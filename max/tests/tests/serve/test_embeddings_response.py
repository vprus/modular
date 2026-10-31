# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #


import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import Request, Response
from fastapi.encoders import jsonable_encoder
from max.serve.router import openai_routes
from max.serve.schemas.openai import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage
from starlette.types import Message


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size,dimensions", [(1, 512), (4, 4096)])
async def test_embeddings_response_preserves_json_contract(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, dimensions: int
) -> None:
    expected = CreateEmbeddingResponse(
        object="list",
        model="embedding-\N{LATIN SMALL LETTER E WITH ACUTE}",
        data=[
            Embedding(
                object="embedding",
                index=index,
                embedding=[
                    (column - dimensions // 2) / dimensions
                    for column in range(dimensions)
                ],
            )
            for index in range(batch_size)
        ],
        usage=Usage(
            prompt_tokens=17 * batch_size, total_tokens=17 * batch_size
        ),
    )
    body = json.dumps(
        {"model": expected.model, "input": ["hello"] * batch_size}
    ).encode()

    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/embeddings",
            "headers": [],
        },
        receive,
    )
    request.state.request_id = "embedding-test"
    request.state.request_timer = SimpleNamespace(start_ns=0)
    monkeypatch.setattr(openai_routes, "get_pipeline", lambda *_: object())
    encode = AsyncMock(return_value=expected)
    monkeypatch.setattr(
        openai_routes.OpenAIEmbeddingsResponseGenerator, "encode", encode
    )

    response = await openai_routes.openai_create_embeddings(request)

    # A Response bypasses FastAPI's per-scalar generic traversal while keeping
    # the previous field names, vector ordering, float values and usage data.
    assert isinstance(response, Response)
    assert response.status_code == 200
    assert response.media_type == "application/json"
    assert json.loads(bytes(response.body)) == jsonable_encoder(expected)
    assert len(encode.call_args.args[0]) == batch_size
