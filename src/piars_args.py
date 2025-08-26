from typing import Optional, Dict, Sequence, Union, List
from dataclasses import dataclass, field
import transformers
import typing

@dataclass
class PIARSArguments:
    target_layers: str = field(metadata={"help": "Layers for Representation. Layers are seperate by `,` eg: `10,12,14,16,18,20` "})
    transform_layers: str = field(metadata={"help": "Layers for Representation. Layers are seperate by `,` eg: `10,12,14,16,18,20` "})
    lorra_alpha: float = field(default=5, metadata={"help": "PIARS Hyperparameter alpha"}) 
    beta: int = field(default=300, metadata={"help": "PIARS Hyperparameter beta for progress calculation"})
    lambda_reg: float = field(default=0.0001, metadata={"help": "L2 regularization coefficient"})
    k_max: int = field(default=5, metadata={"help": "Maximum number of conversation turns"})
    trainsets: typing.List[str] = field(default=None, metadata={"help": "A list of trainsets for finetuning the corresponding Concepts/Functions, separated by # for commandline inputs (eg: ['AlpacaSupervisedDataset'])"})
    valsets: typing.List[str] = field(default=None, metadata={"help": "A list of valsets for finetuning the corresponding Concepts/Functions, separated by # for commandline inputs (eg: ['AlpacaSupervisedDataset'])"})
    adv_string: str = field(default="", metadata={"help": "Adversarial string for harmful prompt. (eg: Start with 'Sure here's')"})
    full_layers: bool = field(default=False, metadata={"help": "Whether to drop not used layer during training"})
    use_refusal_retain : Optional[bool] = field(default=True, metadata={"help":"Whether to train on refusal retain set for Llama models"})
    boundary_data_size : int = field(default=0, metadata={"help":"The size of boundary set"})
    multi_turn_data_path : str = field(default=None, metadata={"help":"The path of multi_turn data"})
    use_warm_up : bool = field(default=False, metadata={"help":"Whether to use the warm up for coefficient of PIARS loss"})
    induction_threshold: float = field(default=0.1, metadata={"help": "Threshold for considering a turn as inductive"})

    use_unlearn_loss: bool = field(default=False, metadata={"help": "Whether to use unlearn loss in multi-turn training. If False, only use separate loss."})
    def to_dict(self):
        return dict( 
            target_layers=self.target_layers, 
            transform_layers=self.transform_layers,
            lorra_alpha=self.lorra_alpha,
            beta=self.beta,
            lambda_reg=self.lambda_reg,
            k_max=self.k_max,
            trainsets=self.trainsets,
            valsets=self.valsets,
            full_layers=self.full_layers,
            induction_threshold=self.induction_threshold,
            use_unlearn_loss=self.use_unlearn_loss
        )

@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Union[List[str], str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    adapter_name_or_path: str = field (
        default=None, metadata={"help": "Adapater name"}
    )
    use_lora: bool = field(
        default=False, metadata={"help": "Use LoRA (default: False)"}
    )
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    grouped_to_max_length: bool = field (
        default=False, metadata={"help": "Group to chunks of max length for pretraining"}
    )
    sc_train_subset : Optional[List[str]] = field(default=None,
                                                  metadata={"help":"subset of the sc train set to train on"})
    log_every : Optional[int] = field(default = 10,
                                      metadata = {"help" : "log loss every log_every steps"})
    sc_train_seq_type : Optional[str] = field(default = 'all_text',
                                              metadata = {"help" : "what portion of the sequence to train on. can be all_text or assistant_response"})
    coeff_schedule : Optional[str] = field(default = 'linear_converge',
                                           metadata = {'help' : 'schedule for the coefficients. can be linear_converge or constant'})
    sc_loss_type : Optional[str] = field(default = 'orig_act_dotprod',
                                         metadata = {'help' : 'type of loss function for shortcircuiting. can be orig_act_dotprod, rand_vec_norm, pos_constant_rmu_{coeff}, center_constant_rmu_{coeff}'})