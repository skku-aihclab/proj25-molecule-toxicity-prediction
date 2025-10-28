import numpy as np

import os
import sys
from example_search_library import *
ROOT = os.path.dirname(  # 1) 스크립트 파일의 폴더: …\CReSS
    os.path.dirname(     # 2) → …\CNN1D
        os.path.dirname( # 3) → …\models
            os.path.dirname(os.path.abspath(__file__))  # 4) → …\my_model
        )
    )
)
if ROOT not in sys.path:
    sys.path.append(ROOT)
    
# 2) 필요한 경로 정의
data_dir   = os.path.join(ROOT, "data")
# (예시) 스펙트럼 npy 파일이 data/spectra/TOX2.npy 라는 이름으로 있다고 가정
sample_npy = os.path.join(data_dir, "spectra", "TOX2.npy")

# ModelInference 의 설정(json/pth) 파일이 있는 폴더(CNN1D) 경로
cnn_dir = os.path.join(ROOT, "models", "CNN1D")

# 3) ModelInference import
from infer import ModelInference

def main():
    # 4) ModelInference 생성
    config_path = os.path.join(cnn_dir, "8.json")
    pretrain_model_path = os.path.join(cnn_dir, "8.pth")
    model_inf = ModelInference(
        config_path=config_path,
        pretrain_model_path=pretrain_model_path,
        device="cpu"
    )

    # 5) 예시 npy 파일 로드: (n_peaks, 2) 형태라고 가정
    arr = np.load(sample_npy)  # e.g. shape=(15, 2)
    # intensity 무시하고 ppm 값만 리스트로 뽑아냄
    ppm_list = arr[:, 0].tolist()

    # 6) NMR 인코딩 수행
    nmr_feature = model_inf.nmr_encode(ppm_list)  # torch.Tensor, shape=(1, 768)
    print("NMR embedding shape:", nmr_feature.shape)
    # 첫 10개 값 예시 출력
    print("First 10 values of embedding:", nmr_feature.flatten()[:10].cpu().numpy())

if __name__ == "__main__":
    main()
#