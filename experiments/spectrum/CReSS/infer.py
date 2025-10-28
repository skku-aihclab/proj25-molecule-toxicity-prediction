import json
import torch
import numpy as np
from CReSS.model.build_model import build_chem_clip_model


class ModelInference(object):
    def __init__(self, config_path, pretrain_model_path, device=None):
        assert (config_path is not None, "config_path is None")
        assert (pretrain_model_path is not None, "pretrain_model_path is None")

        # 1) 디바이스 설정 (cpu/cuda)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 2) 설정 JSON 불러오기
        with open(config_path, "r") as f:
            self.config_json = json.loads(f.read())

        # 3) Chem-CLIP 모델(스펙트럼+SMILES 전부 포함된 모델) 생성
        #    → SMILES 인코더는 당장 필요 없으므로, infer.py 내부에서 사용하지 않습니다.
        self.clip_model = build_chem_clip_model(**self.config_json["model_config"])
        #    build_chem_clip_model 내부에서 NMR과 SMILES 네트워크 모두 생성됨

        # 4) checkpoint에서 weight 로드
        self.clip_model.load_weights(pretrain_model_path)
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()  # 평가 모드로 전환

    def nmr_encode(self, nmr_list):
        """
        nmr_list: List[float], 예) [17.7, 20.0, 22.9, ...]
        내부적으로 nmr2tensor()를 거쳐 (1×(units)) 형태의 binary vector로 만들어 준 뒤
        clip_model.nmr_model.encode()를 호출합니다.
        """
        # ── 이미 4000-dim binary tensor를 넘겨받았으면, 바로 encode
        if isinstance(nmr_list, torch.Tensor):
            # single vector (units,) → (1, units)
            if nmr_list.dim() == 1:
                nmr_tensor = nmr_list.view(1, -1).to(self.device)
            else:
                nmr_tensor = nmr_list.to(self.device)
            with torch.no_grad():
                emb = self.clip_model.nmr_model.encode(nmr_tensor)   # (B, 768)
                emb = self.clip_model.norm_feature(emb)               # (B, 768)
            return emb

        with torch.no_grad():
            # 단일 리스트인지, 배치인지 구분
            if not isinstance(nmr_list[0], list):
                # 단일 입력
                nmr_tensor = self.nmr2tensor(nmr_list).to(self.device)  # shape=(units,)
                # clip_model.nmr_model.encode는 (B, units) 형태를 받으므로 view
                nmr_tensor = nmr_tensor.view(1, -1)                    # (1, units)
                emb = self.clip_model.nmr_model.encode(nmr_tensor)     # (1, 768)
                emb = self.clip_model.norm_feature(emb)                 # 정규화
                return emb                                              # (1, 768)

            else:
                # 배치 입력: List[List[float]]
                tensors = [self.nmr2tensor(x).to(self.device) for x in nmr_list]  # [(units,), ...]
                batch_tensor = torch.stack(tensors, dim=0)                        # (B, units)
                emb = self.clip_model.nmr_model.encode(batch_tensor)              # (B, 768)
                emb = self.clip_model.norm_feature(emb)                            # (B, 768)
                return emb

    def nmr2tensor(self, nmr, scale=10, min_value=-50, max_value=350):
        """
        CReSS 논문에서 제안한 방식대로,
        -50.0 ~ 350.0 ppm 구간을 scale 단위로 나눠서(예: 10 → 0.1 ppm당 1 스텝),
        해당 index에 1을 세팅하는 binary 벡터 생성.
        """
        units = int((max_value - min_value) * scale)  # 예: (350 - -50) * 10 = 4000
        item = np.zeros(units, dtype=np.float32)      # 초기화
        # ppm 값 리스트를 0..units-1 인덱스 값으로 변환
        idxs = [round((val - min_value) * scale) for val in nmr]
        for idx in idxs:
            if idx < 0:
                item[0] = 1.0
            elif idx >= units:
                item[-1] = 1.0
            else:
                item[idx] = 1.0

        return torch.from_numpy(item)  # (units,) tensor, dtype=torch.float32

