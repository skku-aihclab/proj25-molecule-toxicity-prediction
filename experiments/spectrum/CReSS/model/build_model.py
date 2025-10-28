from CReSS.model.model_clip import *
from CReSS.model.model_nmr import *
from CReSS.model.model_smiles import *


def build_chem_clip_model(
    roberta_model_path=None,  # SMILES 모델 경로 (더 이상 사용하지 않음)
    roberta_tokenizer_path=None,  # SMILES 토크나이저 경로 (더 이상 사용하지 않음)
    smiles_maxlen=300,  # SMILES 관련 파라미터 (사용하지 않음)
    vocab_size=55,      # SMILES 관련 파라미터 (사용하지 않음)
    max_position_embeddings=300,  # SMILES 관련 파라미터 (사용하지 않음)
    num_attention_heads=12,  # SMILES 관련 파라미터 (사용하지 않음)
    num_hidden_layers=3,     # SMILES 관련 파라미터 (사용하지 않음)
    type_vocab_size=1,       # SMILES 관련 파라미터 (사용하지 않음)
    nmr_input_channels=400,  # NMR 입력 채널 (예: 4000-length binary vector라면 4000)
    nmr_model_fn=None,       # NMR 모델 구성 함수 (필요 시 전달)
    frozen_smiles_model=False,  # SMILES 모델 동결 여부 (사용하지 않음)
    frozen_nmr_model=False,     # NMR 모델 동결 여부
    flag_use_middleware=False,   # middleware 사용 여부 (옵션)
    loss_fn=None,               # loss 함수 지정 (옵션)
    smile_use_tanh=False,       # SMILES 활성화 함수 여부 (사용하지 않음)
    flag_use_big_class=False,   # big class 사용 여부 (옵션)
    class_number=130000,        # 클래스 수 (옵션)
    feature_dim=768             # 최종 임베딩 차원 (기본 768)
):
    """
    SMILES 인코더 부분을 제거하고, 오직 NMR 인코더만 생성합니다.
    기존에 SmilesModel을 만들던 코드를 삭제하고 smiles_model=None으로 설정합니다.
    NMR 모델의 출력 채널(nmr_output_channels)은 feature_dim으로 지정합니다.
    """

    # 1) feature_dim 유효성 검사
    if feature_dim is None or not isinstance(feature_dim, int) or feature_dim <= 0:
        feature_dim = 768

    # 2) SMILES 모델 생성 부분을 건너뛴다 → smiles_model=None
    smiles_model = None

    # 3) NMR 모델 생성
    #    - input_channels: nmr_input_channels (예: 4000)
    #    - output_channels: feature_dim (이제 SMILES hidden size가 아니라 feature_dim으로 맞춤)
    nmr_model = NMRModel(
        input_channels=nmr_input_channels,
        nmr_output_channels=feature_dim,
        model_fn=nmr_model_fn
    )

    # 4) ChemCLIP 모델에 SMILES는 None, NMR 모델만 넣어준다.
    clip_model = ChemClipModel(
        smiles_model=None,
        nmr_model=nmr_model,
        frozen_smiles_model=frozen_smiles_model,  # 의미는 없어졌지만 인자 형태를 유지
        frozen_nmr_model=frozen_nmr_model,
        flag_use_middleware=flag_use_middleware,
        loss_fn=loss_fn,
        flag_use_big_class=flag_use_big_class,
        class_number=class_number,
        feature_dim=feature_dim
    )

    return clip_model