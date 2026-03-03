from dataclasses import dataclass, field

@dataclass
class LossWeights:
    # Only adversarial domain loss weight (L_total = L_pcr + adv * L_adv)
    adv: float = 0.05


@dataclass
class ModelConfig:
    # dims
    d: int = 256
    dz: int = 128

    num_node_types: int = 3
    num_relations: int = 7

    # gnn
    rgcn_layers: int = 2
    basis: int = 8
    use_attn_pool: bool = False

    # GRL
    grl_lambda: float = 1.0

    # DCE image
    # NOTE: used as in_channels for 3D CNN, should match input x_dce[:, C, D, H, W]
    dce_timepoints: int = 3
    spatial_dims: int = 3
    use_monai: bool = True  # set False if you prefer the lightweight CNN in encoders.py

    # extra edges (kept for compatibility; model_updated currently uses full connectivity for tab<->tab)
    add_extra_edges: bool = False
    extra_topk: int = 3



@dataclass
class TrainConfig:
    # data
    csv_path: str = "PATH/TO/DATA.csv"
    # e.g. {"Center1": "/data/center1", "Center2": "/data/center2", ...}
    roots: dict = None

    # optimization
    epochs: int = 50
    batch_size: int = 16
    lr: float = 6e-4
    weight_decay: float = 1e-4

    loss_w: LossWeights = field(default_factory=LossWeights)

    # system
    num_workers: int = 1
    pin_memory: bool = True
    amp: bool = True

    # logging / ckpt
    log_every: int = 50
    save_path: str =  ""
