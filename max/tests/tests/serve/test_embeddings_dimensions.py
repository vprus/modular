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

import time
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi import FastAPI, Request, Response
from httpx import ASGITransport, AsyncClient
from max.pipelines.modeling.types import EmbeddingsGenerationOutput
from max.serve.router import openai_routes
from transformers import PretrainedConfig


@pytest.fixture
def endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FastAPI, SimpleNamespace, SimpleNamespace]:
    config = SimpleNamespace(
        model=SimpleNamespace(
            pool_embeddings=True,
            huggingface_config=PretrainedConfig(
                hidden_size=4096, is_matryoshka=True
            ),
        )
    )
    rng = np.random.default_rng(57)
    vectors = rng.normal(size=(2, 4096)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    outputs = {
        name: EmbeddingsGenerationOutput(vector)
        for name, vector in zip(("first", "second"), vectors, strict=True)
    }
    pipeline = SimpleNamespace(
        model_name="matryoshka-test",
        encode=AsyncMock(side_effect=lambda request: outputs[request.prompt]),
        outputs=outputs,
    )
    monkeypatch.setattr(openai_routes, "get_pipeline", lambda *_: pipeline)
    monkeypatch.setattr(
        openai_routes, "get_app_pipeline_config", lambda _: config
    )
    app = FastAPI()
    app.add_api_route(
        "/v1/embeddings",
        openai_routes.openai_create_embeddings,
        methods=["POST"],
        response_model=None,
    )

    @app.middleware("http")
    async def timer(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = "embedding-test"
        request.state.request_timer = SimpleNamespace(
            start_ns=time.perf_counter_ns()
        )
        return await call_next(request)

    return app, pipeline, config


async def post(app: FastAPI, **fields: Any) -> Any:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post(
            "/v1/embeddings",
            json={"model": "matryoshka-test", "input": "first", **fields},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("dimensions", [128, 512, 4096])
@pytest.mark.parametrize("batch_size", [1, 2])
async def test_dimensions_crop_then_normalize(
    endpoint: tuple[FastAPI, SimpleNamespace, SimpleNamespace],
    dimensions: int,
    batch_size: int,
) -> None:
    app, pipeline, _ = endpoint
    names = ["first", "second"][:batch_size]
    originals = [pipeline.outputs[name].embeddings.copy() for name in names]
    response = await post(app, input=names, dimensions=dimensions)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "matryoshka-test"
    assert len(body["data"]) == batch_size
    for index, (item, original) in enumerate(
        zip(body["data"], originals, strict=True)
    ):
        actual = np.asarray(item["embedding"])
        expected = original[:dimensions].astype(np.float64)
        expected /= np.linalg.norm(expected)
        assert item["index"] == index
        assert actual.shape == (dimensions,)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)
        assert abs(np.linalg.norm(actual) - 1) < 2e-6
        # Request formatting must not mutate a shared worker result.
        np.testing.assert_array_equal(
            pipeline.outputs[names[index]].embeddings, original
        )
    assert body["usage"] == {"prompt_tokens": 0, "total_tokens": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("fields", [{}, {"dimensions": None}])
async def test_omitted_dimensions_preserve_full_output(
    endpoint: tuple[FastAPI, SimpleNamespace, SimpleNamespace],
    fields: dict[str, Any],
) -> None:
    app, pipeline, config = endpoint
    config.model.huggingface_config.is_matryoshka = False
    # Also preserve the previous behavior for an unnormalized model result.
    pipeline.outputs["first"].embeddings *= 3
    response = await post(app, **fields)
    assert response.status_code == 200
    np.testing.assert_array_equal(
        response.json()["data"][0]["embedding"],
        pipeline.outputs["first"].embeddings,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("dimensions", [0, -1, 4097, True, 512.0, "512"])
async def test_invalid_dimensions_return_400_before_inference(
    endpoint: tuple[FastAPI, SimpleNamespace, SimpleNamespace],
    dimensions: Any,
) -> None:
    app, pipeline, _ = endpoint
    response = await post(app, dimensions=dimensions)
    assert response.status_code == 400
    pipeline.encode.assert_not_awaited()
    # A rejected request must not poison the endpoint.
    assert (await post(app, dimensions=512)).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("pooled,matryoshka", [(True, False), (False, True)])
async def test_model_must_support_pooled_matryoshka_embeddings(
    endpoint: tuple[FastAPI, SimpleNamespace, SimpleNamespace],
    pooled: bool,
    matryoshka: bool,
) -> None:
    app, pipeline, config = endpoint
    config.model.pool_embeddings = pooled
    config.model.huggingface_config.is_matryoshka = matryoshka
    response = await post(app, dimensions=512)
    assert response.status_code == 400
    assert "Matryoshka" in response.json()["detail"]
    pipeline.encode.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_can_declare_allowed_dimensions(
    endpoint: tuple[FastAPI, SimpleNamespace, SimpleNamespace],
) -> None:
    app, pipeline, config = endpoint
    config.model.huggingface_config.is_matryoshka = False
    config.model.huggingface_config.matryoshka_dimensions = [128, 512, 4096]
    assert (await post(app, dimensions=256)).status_code == 400
    pipeline.encode.assert_not_awaited()
    assert (await post(app, dimensions=512)).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
async def test_invalid_truncated_vector_returns_500(
    endpoint: tuple[FastAPI, SimpleNamespace, SimpleNamespace],
    value: float,
) -> None:
    app, pipeline, _ = endpoint
    pipeline.outputs["first"].embeddings[:512] = value
    assert (await post(app, dimensions=512)).status_code == 500
    assert (await post(app, input="second", dimensions=512)).status_code == 200


@pytest.mark.asyncio
async def test_dimensions_cannot_exceed_actual_worker_output(
    endpoint: tuple[FastAPI, SimpleNamespace, SimpleNamespace],
) -> None:
    app, pipeline, _ = endpoint
    pipeline.outputs["first"].embeddings = np.ones(128, dtype=np.float32)
    assert (await post(app, dimensions=512)).status_code == 400
