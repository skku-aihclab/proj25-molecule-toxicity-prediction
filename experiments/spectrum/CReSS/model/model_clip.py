import torch
from torch import nn
from CReSS.model.model import SelfCorrelationBaseModel


class ChemClipModel(SelfCorrelationBaseModel):
    def __init__(self,
                 smiles_model=None,
                 nmr_model=None,
                 frozen_smiles_model=False,
                 frozen_nmr_model=False,
                 flag_use_middleware=False,
                 loss_fn=None,
                 flag_use_big_class=False,
                 class_number=130000,
                 feature_dim=768):
        super(ChemClipModel,
              self).__init__(loss_fn=loss_fn,
                             flag_use_big_class=flag_use_big_class,
                             class_number=class_number)

        self.smiles_model = smiles_model
        self.nmr_model = nmr_model
        self.frozen_smiles_model = frozen_smiles_model
        self.frozen_nmr_model = frozen_nmr_model
        self.flag_use_middleware = flag_use_middleware
        self.feature_dim = feature_dim

        if self.flag_use_middleware:
            self.smiles_middleware = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim))
            self.nmr_middleware = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim))

        # ── SMILES 모델이 None이 아닐 때만 train/eval 호출 ──
        if self.smiles_model is not None:
            if self.frozen_smiles_model:
                self.smiles_model.eval()
            else:
                self.smiles_model.train()
        # smiles_model이 None이면 아무 작업도 하지 않음

        # ── NMR 모델도 None인지 확인하고 train/eval 호출 ──
        if self.nmr_model is not None:
            if self.frozen_nmr_model:
                self.nmr_model.eval()
            else:
                self.nmr_model.train()
        # nmr_model이 None이면 아무 작업도 하지 않음

    def load_weights(self, path):
        if path is None:
            return

        # 1) 전체 체크포인트 불러오기
        ckpt = torch.load(path, map_location=torch.device('cpu'))

        # 2) 'nmr_model.' 으로 시작하는 키들만 필터링
        nmr_state_dict = {}
        for key, value in ckpt.items():
            # 예: key == "nmr_model.layer1.weight" 등
            if key.startswith("nmr_model."):
                # 'nmr_model.' 다음 부분을 그대로 사용
                new_key = key[len("nmr_model."):]
                nmr_state_dict[new_key] = value

        # 3) NMR 모델 가중치 로드
        if self.nmr_model is not None:
            self.nmr_model.load_state_dict(nmr_state_dict)
        else:
            raise RuntimeError("NMR model이 정의되지 않은 상태에서 체크포인트를 로드하려 합니다.")
    def smiles_model_eval(self):
        self.frozen_smiles_model = True
        if self.smiles_model is not None:
            self.smiles_model.eval()

    def smiles_model_train(self):
        self.frozen_smiles_model = False
        if self.smiles_model is not None:
            self.smiles_model.train()

    def nmr_model_eval(self):
        self.frozen_nmr_model = True
        if self.nmr_model is not None:
            self.nmr_model.eval()

    def nmr_model_train(self):
        self.frozen_nmr_model = False
        if self.nmr_model is not None:
            self.nmr_model.train()
