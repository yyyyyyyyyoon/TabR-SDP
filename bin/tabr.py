import pandas as pd
import os

if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import math
from dataclasses import dataclass
from typing import Literal, Optional, Union
from sklearn.model_selection import KFold

import delu
import faiss
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
from torch import Tensor
import lib
from data import preprocess_data
import torch
from sklearn.metrics import confusion_matrix
from math import sqrt
from typing import Dict, Any
import json

KWArgs = Dict[str, Any]
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

@dataclass(frozen=True)
class Config:
    seed: int
    data: Dict[str, numpy.ndarray]
    model: KWArgs  # Model
    context_size: int
    optimizer: KWArgs  # lib.deep.make_optimizer
    batch_size: int
    patience: Optional[int]
    n_epochs: Union[int, float]


class Model(nn.Module):
    def __init__(
        self,
        *,
        #
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: Optional[int],
        #
        num_embeddings: Optional[dict],  # lib.deep.ModuleSpec
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        predictor_n_blocks: int,
        mixer_normalization: Union[bool, Literal['auto']],
        context_dropout: float,
        dropout0: float,
        dropout1: Union[float, Literal['dropout0']],
        normalization: str,
        activation: str,
        distance_metric="IP",
        #
        # The following options should be used only when truly needed.
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
    ) -> None:
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        super().__init__()

        self.distance_metric = distance_metric  # 거리 계산 방식 변수 설정
        self.search_index = None  # 검색 인덱스 초기화

        # normalization 클래스 가져오기
        Normalization = getattr(nn, normalization)
        # activation 클래스 가져오기
        Activation = getattr(nn, activation)

        if dropout1 == 'dropout0':
            dropout1 = dropout0

        self.one_hot_encoder = (
            lib.OneHotEncoder(cat_cardinalities) if cat_cardinalities else None
        )
        self.num_embeddings = (
            None
            if num_embeddings is None
            else lib.make_module(num_embeddings, n_features=n_num_features)
        )

        # >>> E
        n_bin_features = 0
        cat_cardinalities = []
        d_in = (
            n_num_features
            + n_bin_features
            + sum(cat_cardinalities)
        )
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        Activation = getattr(nn, activation)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [make_block(i > 0) for i in range(encoder_n_blocks)]
        )

        # >>> R
        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes is None
            else nn.Sequential(
                nn.Embedding(n_classes, d_main), delu.nn.Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        # >>> P
        self.blocks1 = nn.ModuleList(
            [make_block(True) for _ in range(predictor_n_blocks)]
        )
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(),
            nn.Linear(d_main, 2), # num_classes=2
        )

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]  # noqa: E501
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]  # noqa: E501
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]  # noqa: E501

    def _encode(self, x_: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x_num = x_.get('num')
        x_bin = x_.get('bin')
        x_cat = x_.get('cat')
        del x_

        x = []
        if x_num is None:
            assert self.num_embeddings is None
        else:
            x.append(
                x_num
                if self.num_embeddings is None
                else self.num_embeddings(x_num).flatten(1)
            )
        if x_bin is not None:
            x.append(x_bin)
        if x_cat is None:
            assert self.one_hot_encoder is None
        else:
            assert self.one_hot_encoder is not None
            x.append(self.one_hot_encoder(x_cat))
        assert x
        x = torch.cat(x, dim=1)
        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    def build_search_index(self, d_main, device):
        # 거리 방식에 따라 FAISS 인덱스 생성
        if self.distance_metric == "L2":
            return (
                faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                if device.type == 'cuda'
                else faiss.IndexFlatL2(d_main)
            )
        elif self.distance_metric == "IP":
            return (
                faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), d_main)
                if device.type == 'cuda'
                else faiss.IndexFlatIP(d_main)
            )
        else:
            raise ValueError(f"지원하지 않는 거리 방식: {self.distance_metric}")

    def compute_similarity(self, k, context_k):
        if self.distance_metric == "L2":
            # L2 거리 기반 유사도
            return (
                    -k.square().sum(-1, keepdim=True)
                    + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
                    - context_k.square().sum(-1)
            )
        elif self.distance_metric == "IP":
            # 내적 기반 유사도 (Cosine 유사도도 정규화 후 여기서 가능)
            return (k[..., None, :] @ context_k.transpose(-1, -2)).squeeze(-2)
        else:
            raise ValueError(f"지원하지 않는 거리 방식: {self.distance_metric}")

    def forward(
        self,
        x_: dict[str, Tensor],
        y: Optional[Tensor],
        candidate_x_: dict[str, Tensor],
        candidate_y: Tensor,
        context_size: int,
        is_train: bool,
    ) -> Tensor:
        # >>>
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.memory_efficient
        ):
            candidate_k = (
                self._encode(candidate_x_)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [
                        self._encode(x)[1]
                        for x in delu.iter_batches(
                            candidate_x_, self.candidate_encoding_batch_size
                        )
                    ]
                )
            )
        x, k = self._encode(x_)
        if is_train:
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None

        batch_size, d_main = k.shape
        device = k.device
        with torch.no_grad():
            if self.search_index is None:
                self.search_index = self.build_search_index(d_main, device) #조건문 기반 인덱스 생성
            self.search_index.reset()
            self.search_index.add(candidate_k)
            distances: Tensor
            context_idx: Tensor
            distances, context_idx = self.search_index.search(
                k, context_size + (1 if is_train else 0)
            ) # L2/IP 동일 방식 사용, 유사도 수식만 다름

            if is_train:
                distances[
                    context_idx == torch.arange(batch_size, device=device)[:, None]
                ] = torch.inf
                top_indices = distances.argsort(dim=1)[:, :-1]
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])
                distances = distances.gather(1, top_indices)

                # 같은 클래스 후보만 필터링
                #batch_size = y.shape[0]
                #expanded_y = y[:, None].expand(-1, context_idx.size(1))
                #selected_candidate_y = candidate_y[context_idx]

                # y가 같은 후보가 아닌 경우 → 거리를 inf로 바꿔서 softmax에서 제외
                #label_mask = (selected_candidate_y == expanded_y)
                #distances[~label_mask] = float('inf')

                # 다시 가장 가까운 context_size개 재정렬
                #context_idx = context_idx.gather(1, distances.argsort(dim=1)[:, :context_size])

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            context_k = self._encode(
                {
                    ftype: torch.cat([x_[ftype], candidate_x_[ftype]])[
                        context_idx
                    ].flatten(0, 1)
                    for ftype in x_
                }
            )[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]

        similarities = self.compute_similarity(k, context_k)
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # >>>
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass(frozen=True)
class Config:
    seed: int
    data: Dict[str, numpy.ndarray]
    model: KWArgs
    context_size: int
    optimizer: KWArgs
    batch_size: int
    patience: Optional[int]
    n_epochs: Union[int, float]

def evaluate(model, X_eval_tensor, y_eval_tensor, candidate_X_tensor, candidate_y_tensor, context_size, eval_batch_size=32):
    model.eval()
    preds = []
    with torch.inference_mode():
        while eval_batch_size:
            try:
                for i in range(0, len(X_eval_tensor), eval_batch_size):
                    batch_X = X_eval_tensor[i:i + eval_batch_size]
                    x_ = {'num': batch_X}
                    candidate_x_ = {'num': candidate_X_tensor}   # 후보 데이터 (예시로 학습 데이터 사용)
                    candidate_y = candidate_y_tensor
                    output = model(
                        x_=x_,
                        y=None,
                        candidate_x_=candidate_x_,
                        candidate_y=candidate_y,
                        context_size=context_size,
                        is_train=False,
                    )
                    preds.append(output.cpu())
            except RuntimeError as err:
                if "out of memory" not in str(err).lower():
                    raise
                eval_batch_size //= 2
            else:
                break

        preds = torch.cat(preds)
        _, y_pred = torch.max(preds, dim=1)
        y_true_np = y_eval_tensor.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        # confusion matrix, f1은 FIR 계산에 필요
        cm = confusion_matrix(y_true_np, y_pred_np)
        TP, FP, FN, TN = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
        FI = (TP + FP) / (TP + FP + TN + FN)
        PD = TP / (TP + FN) if (TP + FN) > 0 else 0
        PF = FP / (FP + TN) if (FP + TN) > 0 else 0
        FIR = (PD - FI) / PD if PD > 0 else 0
        Balance = 1 - (sqrt((0 - PF) ** 2 + (1 - PD) ** 2) / sqrt(2))

        return {'PD': PD, 'PF': PF, 'FIR': FIR, 'Balance': Balance}

def load_best_params(result_json_path):
    with open(result_json_path, "r") as f:
        result = json.load(f)
    params = result.get("params", {})
    # 필요한 파라미터 모두 추출
    n_epochs = params.get("n_epochs")
    d_main = params.get("d_main")
    encoder_n_blocks = params.get("encoder_n_blocks")
    context_size = params.get("context_size")
    predictor_n_blocks = params.get("predictor_n_blocks")
    return n_epochs, d_main, encoder_n_blocks, context_size, predictor_n_blocks

def run_experiment_for_dataset(csv_path, param_json_path):
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    n_epochs, d_main, encoder_n_blocks, context_size, predictor_n_blocks = load_best_params(param_json_path)
    splits = preprocess_data(csv_path)
    if dataset_name not in splits:
        print(f"Error: Dataset '{dataset_name}' not found in splits.")
        return None

    X_all = numpy.array(splits[dataset_name]["X_train"])
    y_all = numpy.array(splits[dataset_name]["y_train"], dtype=int)

    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        print(f"\n[Fold {fold + 1}/{n_splits}]")

        # Fold별 데이터 나누기
        X_tr, X_te =  X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        # Tensor로 변환
        X_tr_tensor = torch.from_numpy(X_tr).float().to(device)
        X_te_tensor = torch.from_numpy(X_te).float().to(device)
        y_tr_tensor = torch.from_numpy(y_tr).long().to(device)
        y_te_tensor = torch.from_numpy(y_te).long().to(device)

        # model 생성
        model = Model(
            n_num_features=X_tr.shape[1],
            n_bin_features=1,
            cat_cardinalities=[],
            n_classes=2,
            num_embeddings=None,  # 또는 필요한 딕셔너리 제공
            d_main=d_main, # JSON의 값
            d_multiplier=2.0,
            encoder_n_blocks=encoder_n_blocks, # JSON의 값
            predictor_n_blocks=predictor_n_blocks, # JSON의 값
            mixer_normalization='auto',
            context_dropout=0.1,
            dropout0=0.2,
            dropout1='dropout0',  # 또는 float 값 제공
            normalization='LayerNorm',
            activation='ReLU',
            memory_efficient=False,
            candidate_encoding_batch_size=None,
            distance_metric="IP"
        ).to(device)

        # 모델 훈련
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(n_epochs):
            model.train()
            for batch_idx in lib.make_random_batches(len(y_tr_tensor), 32, device):
                batch_x =  X_tr_tensor[batch_idx]
                batch_y = y_tr_tensor[batch_idx]
                x_ = {'num': batch_x}
                candidate_x_ = {'num': X_tr_tensor}
                candidate_y = y_tr_tensor

                optimizer.zero_grad()
                output = model(x_, batch_y, candidate_x_, candidate_y, context_size=context_size, is_train=True)
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()

        # fold별 검증 결과 저장
        metrics = evaluate(model, X_te_tensor, y_te_tensor, X_tr_tensor, y_tr_tensor, context_size=context_size)
        all_metrics.append(metrics)

    avg = pd.DataFrame(all_metrics).mean()

    # fold별 마지막 검증 결과 평균 출력
    print(f"== [{dataset_name}] 10-Fold 평균성능 ==")
    print(avg)
    return avg


def main() -> None:
    data_dir = "data"
    result_dir = "results"
    all_results = {}

    # 특정 CSV 선택
    if len(sys.argv) > 1:
        target_csv = os.path.basename(sys.argv[1])
        print(f"[INFO] 선택된 데이터셋: {target_csv}")
    else:
        target_csv = None

    for fname in os.listdir(data_dir):
        if not fname.endswith(".csv"):
            continue
        # 특정 CSV 실행
        if target_csv is not None and fname != target_csv:
            continue

        csv_path = os.path.join(data_dir, fname)
        dataset_name = os.path.splitext(fname)[0]
        param_json_path = os.path.join(result_dir, f"{dataset_name}_hyperparameter.json")

        if not os.path.exists(param_json_path):
            print(f"[SKIP] '{dataset_name}' 하이퍼파라미터 JSON 없음")
            continue

        print(f"\n=== {dataset_name} 실험 시작 ===")
        avg_metrics = run_experiment_for_dataset(csv_path, param_json_path)

        if avg_metrics is not None:
            all_results[dataset_name] = avg_metrics

    result_df = pd.DataFrame(all_results).T
    print(result_df)

    # 결과 저장
    result_df.to_csv("results/all_results_IP.csv")

if __name__ == '__main__':
    main()