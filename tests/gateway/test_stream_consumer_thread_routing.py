"""Regression tests for stream consumer thread/topic routing fix.

Verifies that GatewayStreamConsumer correctly passes reply_to on the first
message send, ensuring messages land in the correct topic/thread instead of
the main group chat.

Covers: #6969, #9916, #7355
"""
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

import pytest

from gateway.stream_consumer import (
    GatewayStreamConsumer,
)


def _make_adapter(send_result=None, edit_result=None, max_length=4096):
    adapter = MagicMock()
    adapter.send = AsyncMock(
        return_value=send_result or SimpleNamespace(success=True, message_id="msg_1")
    )
    adapter.edit_message = AsyncMock(
        return_value=edit_result or SimpleNamespace(success=True)
    )
    adapter.MAX_MESSAGE_LENGTH = max_length
    return adapter


class TestInitialReplyToId:
    """Verify initial_reply_to_id is passed as reply_to on first send."""

    @pytest.mark.asyncio
    async def test_first_send_uses_initial_reply_to_id(self):
        """When initial_reply_to_id is set, first adapter.send() should
        include reply_to=initial_reply_to_id."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            metadata={"thread_id": "omt_topic123"},
            initial_reply_to_id="om_user_msg_456",
        )
        await consumer._send_or_edit("Hello world")

        adapter.send.assert_called_once()
        call_kwargs = adapter.send.call_args[1]
        assert call_kwargs["reply_to"] == "om_user_msg_456", (
            "First send should pass initial_reply_to_id as reply_to"
        )
        assert call_kwargs["chat_id"] == "chat_123"


    @pytest.mark.asyncio
    async def test_subsequent_edits_ignore_initial_reply_to_id(self):
        """After first send, edits should use message_id, not initial_reply_to_id."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            metadata={"thread_id": "omt_topic123"},
            initial_reply_to_id="om_user_msg_456",
        )

        # First send
        await consumer._send_or_edit("Hello world")
        assert adapter.send.call_count == 1

        # Second call should edit, not send
        await consumer._send_or_edit("Hello world updated")
        assert adapter.send.call_count == 1, "Should edit, not send again"
        adapter.edit_message.assert_called_once()
        edit_kwargs = adapter.edit_message.call_args[1]
        assert edit_kwargs["message_id"] == "msg_1"
        assert edit_kwargs["chat_id"] == "chat_123"


class TestOverflowFirstMessage:
    """Verify thread routing is preserved when the first message overflows."""

    @pytest.mark.asyncio
    async def test_overflow_first_send_uses_initial_reply_to_id(self):
        """When first message exceeds platform limit and is split into chunks,
        each chunk should be threaded to initial_reply_to_id, not None."""
        adapter = _make_adapter(max_length=10)
        adapter.truncate_message = MagicMock(
            return_value=["chunk_1", "chunk_2"]
        )
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            metadata={"thread_id": "omt_topic123"},
            initial_reply_to_id="om_user_msg_789",
        )

        # Inject oversized accumulated text to trigger overflow path
        consumer._accumulated = "A" * 100
        consumer._current_edit_interval = 999
        await consumer._send_new_chunk("chunk_1", consumer._message_id or consumer._initial_reply_to_id)

        adapter.send.assert_called_once()
        call_kwargs = adapter.send.call_args[1]
        assert call_kwargs["reply_to"] == "om_user_msg_789", (
            "Overflow first chunk should use initial_reply_to_id"
        )


class TestFeishuFallbackThreadRouting:
    """Verify FeishuAdapter._send_raw_message keeps an anchorless topic send in its topic."""

    def _wire(self, anchor):
        """A spec'd adapter with the real builders and a mocked Lark client."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = MagicMock(spec=FeishuAdapter)
        client = MagicMock()
        ok = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="new_msg_1"))
        client.im.v1.message.create = MagicMock(return_value=ok)
        client.im.v1.message.reply = MagicMock(return_value=ok)

        adapter._client = client
        adapter._build_create_message_body = FeishuAdapter._build_create_message_body
        adapter._build_create_message_request = FeishuAdapter._build_create_message_request
        adapter._build_reply_message_body = FeishuAdapter._build_reply_message_body
        adapter._build_reply_message_request = FeishuAdapter._build_reply_message_request
        adapter._fetch_last_message_in_thread = AsyncMock(return_value=anchor)
        # Static/instance helpers the code under test actually relies on: a bare
        # MagicMock(spec=...) would auto-mock them and swallow the real behaviour
        # (_response_succeeded would return a truthy mock, hiding every failure path).
        adapter._response_succeeded = FeishuAdapter._response_succeeded

        async def _reply_in_thread(anchor_message_id, *, msg_type, payload):
            return await FeishuAdapter._reply_in_thread(
                adapter, anchor_message_id, msg_type=msg_type, payload=payload,
            )
        adapter._reply_in_thread = _reply_in_thread

        # _send_raw_message routes blocking SDK calls through _run_blocking
        # (adapter-owned executor). On a MagicMock(spec=...) that method is
        # auto-mocked and would swallow the real call, so wire a passthrough.
        async def _run_blocking_passthrough(func, *args):
            return func(*args)
        adapter._run_blocking = _run_blocking_passthrough
        return adapter, client

    @pytest.mark.asyncio
    async def test_anchorless_topic_send_replies_into_the_topic(self):
        """A topic send with no reply anchor must go through the reply API anchored on the
        topic's latest message: the create API rejects receive_id_type='thread_id' with
        99992402 field-validation-failed, silently dropping the message."""
        import json
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter, client = self._wire("om_topic_anchor")
        result = await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload=json.dumps({"text": "hello"}),
            reply_to=None,
            metadata={"thread_id": "omt_topic_abc"},
        )

        assert result is not None
        client.im.v1.message.reply.assert_called_once()
        assert client.im.v1.message.create.call_count == 0, (
            "receive_id_type='thread_id' is not a valid create parameter (99992402)"
        )
        request = client.im.v1.message.reply.call_args[0][0]
        assert getattr(request, "message_id", None) == "om_topic_anchor"
        body = getattr(request, "body", None) or getattr(request, "request_body", None)
        assert body is not None, "request has neither .body nor .request_body"
        assert getattr(body, "reply_in_thread", None) is True

    @pytest.mark.asyncio
    async def test_anchorless_topic_send_without_anchor_falls_back_to_chat(self):
        """When the topic has no readable anchor, send to the chat instead of dropping it."""
        import json
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter, client = self._wire(None)
        await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload=json.dumps({"text": "hello"}),
            reply_to=None,
            metadata={"thread_id": "omt_topic_abc"},
        )

        client.im.v1.message.reply.assert_not_called()
        client.im.v1.message.create.assert_called_once()
        call_args = client.im.v1.message.create.call_args[0][0]
        # The Lark SDK builder exposes .body; the in-tree fallback exposes .request_body.
        body = getattr(call_args, "body", None) or getattr(call_args, "request_body", None)
        assert body is not None, "request has neither .body nor .request_body"
        receive_id = getattr(body, "receive_id", None)
        if receive_id is None and isinstance(body, str):
            receive_id = json.loads(body).get("receive_id")
        assert receive_id == "oc_main_chat"
        assert getattr(call_args, "receive_id_type", None) == "chat_id"


    @pytest.mark.asyncio
    async def test_withdrawn_anchor_is_replaced_and_retried(self):
        """An anchor withdrawn between the lookup and the reply (230011) must be replaced by
        the next newest message and retried, not dropped."""
        import json
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter, client = self._wire("om_stale")
        adapter._fetch_last_message_in_thread = AsyncMock(side_effect=["om_stale", "om_fresh"])
        client.im.v1.message.reply = MagicMock(side_effect=[
            SimpleNamespace(success=lambda: False, code=230011),  # anchor withdrawn
            SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="new_msg_2")),
        ])

        result = await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload=json.dumps({"text": "hello"}),
            reply_to=None,
            metadata={"thread_id": "omt_topic_abc"},
        )

        assert result is not None
        anchors = [call[0][0].message_id for call in client.im.v1.message.reply.call_args_list]
        assert anchors == ["om_stale", "om_fresh"], "must retry on a fresh anchor"
        assert client.im.v1.message.create.call_count == 0
        reanchor_kwargs = adapter._fetch_last_message_in_thread.await_args_list[1].kwargs
        assert reanchor_kwargs.get("exclude") == "om_stale", "must skip the anchor that failed"

    @pytest.mark.asyncio
    async def test_unrecoverable_anchor_reply_still_reaches_the_chat(self):
        """When even the re-anchored reply is rejected, send to the chat: a message in the
        wrong place beats the silent drop this path used to produce."""
        import json
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter, client = self._wire("om_stale")
        adapter._fetch_last_message_in_thread = AsyncMock(side_effect=["om_stale", None])
        client.im.v1.message.reply = MagicMock(
            return_value=SimpleNamespace(success=lambda: False, code=230011)
        )

        await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload=json.dumps({"text": "hello"}),
            reply_to=None,
            metadata={"thread_id": "omt_topic_abc"},
        )

        assert client.im.v1.message.reply.call_count == 1, "no anchor left to retry on"
        client.im.v1.message.create.assert_called_once()


class TestTopicAnchorLookup:
    """The anchor lookup must return the topic's NEWEST message and be able to skip one."""

    def _wire(self, items):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = MagicMock(spec=FeishuAdapter)
        client = MagicMock()
        client.im.v1.message.list = MagicMock(
            return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(items=items))
        )
        adapter._client = client
        adapter._response_succeeded = FeishuAdapter._response_succeeded

        async def _run_blocking_passthrough(func, *args):
            return func(*args)
        adapter._run_blocking = _run_blocking_passthrough
        return adapter, client

    @pytest.mark.asyncio
    async def test_lookup_asks_for_the_newest_message(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        items = [SimpleNamespace(message_id="om_new"), SimpleNamespace(message_id="om_old")]
        adapter, client = self._wire(items)

        anchor = await FeishuAdapter._fetch_last_message_in_thread(adapter, "omt_topic_abc")

        assert anchor == "om_new"
        request = client.im.v1.message.list.call_args[0][0]
        assert getattr(request, "sort_type", None) == "ByCreateTimeDesc", (
            "the list API defaults to ascending, which returned the topic's OLDEST message"
        )
        assert getattr(request, "page_size", None) == 1
        assert getattr(request, "container_id_type", None) == "thread"
        assert getattr(request, "container_id", None) == "omt_topic_abc"

    @pytest.mark.asyncio
    async def test_lookup_can_skip_a_known_bad_anchor(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        items = [SimpleNamespace(message_id="om_stale"), SimpleNamespace(message_id="om_fresh")]
        adapter, client = self._wire(items)

        anchor = await FeishuAdapter._fetch_last_message_in_thread(
            adapter, "omt_topic_abc", exclude="om_stale"
        )

        assert anchor == "om_fresh"
        assert getattr(client.im.v1.message.list.call_args[0][0], "page_size", None) == 5, (
            "re-anchoring needs more than one candidate to skip past the bad anchor"
        )

    @pytest.mark.asyncio
    async def test_lookup_returns_none_when_every_candidate_is_excluded(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter, _ = self._wire([SimpleNamespace(message_id="om_stale")])
        assert await FeishuAdapter._fetch_last_message_in_thread(
            adapter, "omt_topic_abc", exclude="om_stale"
        ) is None


class TestFallbackResendThreading:
    """#103068: a full fallback resend replaces the preview, so it must land in
    the originating thread; tail continuations keep their unthreaded delivery."""

    @pytest.mark.asyncio
    async def test_full_fallback_resend_threads_first_chunk_only(self):
        adapter = _make_adapter(max_length=700)
        adapter.send.side_effect = [
            SimpleNamespace(success=True, message_id=f"full_{i}") for i in range(10)
        ]
        consumer = GatewayStreamConsumer(adapter, "chat_123", initial_reply_to_id="om_user_1")
        consumer._message_id = "om_preview"
        consumer._last_sent_text = "truncated snapshot"
        consumer._already_sent = True
        consumer._fallback_final_send = True

        final = " ".join(["word"] * 400)  # not prefixed by the snapshot -> full resend
        await consumer._send_fallback_final(final)

        calls = adapter.send.await_args_list
        assert len(calls) > 1
        assert calls[0].kwargs["reply_to"] == "om_user_1"
        assert all("reply_to" not in c.kwargs for c in calls[1:])
