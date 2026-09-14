import argparse
import pprint
import torch
import random
import numpy as np
import os
from datetime import datetime
import logging


from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

supported_models = [
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-13b-hf',
            'meta-llama/Llama-2-70b-hf',
            'meta-llama/Meta-Llama-3-8B',
            'meta-llama/Meta-Llama-3-70B',
            'meta-llama/Meta-Llama-3.1-405B',
            'facebook/opt-125m',
            'Qwen/Qwen3-1.7B',
            'Qwen/Qwen3-8B',
            ]
supported_datasets = ['wikitext2', 'ptb', 'c4']

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def llama_down_proj_groupsize(model, groupsize):
    
    assert groupsize > 1, 'groupsize should be greater than 1!'
    
    if model.config.intermediate_size % groupsize == 0:
        logging.info(f'(Act.) Groupsiz = Down_proj Groupsize: {groupsize}')
        return groupsize

    group_num = int(model.config.hidden_size/groupsize)
    assert groupsize*group_num == model.config.hidden_size, 'Invalid groupsize for llama!'

    down_proj_groupsize = model.config.intermediate_size//group_num
    assert down_proj_groupsize*group_num == model.config.intermediate_size, 'Invalid groupsize for down_proj!'
    logging.info(f'(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}')
    return down_proj_groupsize



def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

# Dump the log both to console and a log file.
def config_logging(log_file, level=logging.INFO):
    class LogFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                self._style._fmt = "%(message)s"
            else:
                self._style._fmt = "%(levelname)s: %(message)s"
            return super().format(record)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LogFormatter())

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(LogFormatter())

    logging.basicConfig(level=level, handlers=[console_handler, file_handler])


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                        help=f'Model name or local path. Examples: {supported_models}')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--eval_dataset', type=str, default='wikitext2',
                        help='Dataset for Evaluation (default: wikitext2)', choices=supported_datasets,)
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--bsz', type=int, default=32,
                        help='Batch-size for PPL evaluation (default:32)')

    # Rotation Arguments
    parser.add_argument('--rotate', action=argparse.BooleanOptionalAction, default=False, 
                        help='''Rotate the moodel. This will include online rotation for down-projection and
                        out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                        if we want to quantize the Keys''')
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    parser.add_argument('--rotation_seed', type=int, default=-1,
                        help='Random Seed for generating random matrix!!')
    parser.add_argument('--fp32_had', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply Hadamard rotation in FP32 (default: False)')

    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)''')
    parser.add_argument('--a_groupsize', type=int, default=-1, 
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize')
    parser.add_argument('--a_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric Activation quantization (default: False)')
    parser.add_argument('--a_clip_ratio', type=float, default=1.0,
        help='Clip ratio for activation quantization. new_max = max * clip_ratio')
    parser.add_argument('--a_per_tensor', action=argparse.BooleanOptionalAction, default=False,
                        help='Per-tensor activation quantization (paper 4.1). Default is per-token.')
    parser.add_argument('--enable_aq_calibration', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable activation quantization in GPTQ(v2) (default: False)')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Local directory for WikiText-2 (save_to_disk or parquet).')

    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16, 
                        help='Number of bits for weights of the Linear layers')
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False,
                        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=False,
                        help='''Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--w_format', type=str, default='int', choices=['int', 'nvfp4'],
                        help='Weight grid: int (affine INT) or nvfp4 (E2M1 + fp8 scale).')
    parser.add_argument('--a_format', type=str, default='int', choices=['int', 'nvfp4'],
                        help='Activation grid: int or nvfp4. W4A4 NVFP4 uses both nvfp4.')
    parser.add_argument('--seqlen', type=int, default=None,
                        help='Override model.seqlen for calibration and PPL (default: model default).')
    parser.add_argument('--eval_nsamples', type=int, default=None,
                        help='Evaluate only the first N WikiText windows (smoke). Default: all.')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default='wikitext2',
                        help='calibration data samples for GPTQ.', choices=supported_datasets)
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False,
                        help='act-order in GPT(A)Q')
    parser.add_argument('--static_groups', action=argparse.BooleanOptionalAction, default=False,
                        help='static groups in GPT(A)Q')
    parser.add_argument('--asym_calibrate', action=argparse.BooleanOptionalAction, default=False,
                        help='enable GPTAQ asymmetric calibration')

    # ReQuant (arXiv:2608.07019) — fixed-grid discrete refinement
    parser.add_argument('--requant', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply ReQuant after the PTQ initializer (Algorithm 1).')
    parser.add_argument('--requant_sweeps', type=int, default=4,
                        help='Number of coordinate sweeps T (paper default: 4).')
    parser.add_argument('--requant_neighborhood', type=int, default=2,
                        help='Neighborhood size K (paper default: 2).')
    parser.add_argument('--requant_fp_branch', type=str, default='true_fp',
                        choices=['true_fp', 'gptaq_local'],
                        help='Source of X in Eq.5. true_fp uses a frozen FP weight snapshot.')
    parser.add_argument('--requant_objective', type=str, default='full',
                        choices=['full', 'simplified'],
                        help='full = Eq.5 ||WX - W_q X~||^2; simplified drops the cross term.')
    parser.add_argument('--requant_cross_alpha', type=float, default=1.0,
                        help='Scale on the Eq.5 cross term (GPTAQ uses 0.25 for its analogue).')
    parser.add_argument('--requant_coord_order', type=str, default='forward',
                        choices=['forward', 'reverse', 'random'],
                        help='Within-row column order. Paper default is forward (j=1..dcol).')
    parser.add_argument('--requant_chunk', type=int, default=32,
                        help='Calibration chunk size for the paired FP / quantized pass.')
    parser.add_argument('--requant_offload_activations', action=argparse.BooleanOptionalAction, default=False,
                        help='Keep inps/fp_inps on CPU and move chunks to GPU on demand.')

    # General Quantization Arguments
    parser.add_argument('--int8_down_proj', action=argparse.BooleanOptionalAction, default=False,
                        help='Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8')

    # KV-Cache Quantization Arguments
    parser.add_argument('--v_bits', type=int, default=16,
                        help='''Number of bits for V-cache quantization. 
                        Note that quantizing the V-cache does not need any other rotation''')
    parser.add_argument('--v_groupsize', type=int, default=-1)
    parser.add_argument('--v_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric V-cache quantization')
    parser.add_argument('--v_clip_ratio', type=float, default=1.0,
        help='Clip ratio for v-cache quantization. new_max = max * clip_ratio')
    
    parser.add_argument('--k_bits', type=int, default=16,
                        help='''Number of bits for K-cache quantization. 
                        Note that quantizing the K-cache needs another rotation for the keys/queries''')
    parser.add_argument('--k_groupsize', type=int, default=-1)
    parser.add_argument('--k_asym', action=argparse.BooleanOptionalAction, default=False, 
                        help='ASymmetric K-cache quantization')
    parser.add_argument('--k_pre_rope', action=argparse.BooleanOptionalAction, default=False, 
                        help='Pre-RoPE quantization for K-cache (not Supported yet!)')
    parser.add_argument('--k_clip_ratio', type=float, default=1.0,
        help='Clip ratio for k-cache quantization. new_max = max * clip_ratio')

    # Save/Load Quantized Model Arguments
    parser.add_argument('--load_qmodel_path', type=str, default=None,
                        help='Load the quantized model from the specified path!')
    parser.add_argument('--save_qmodel_path', type=str, default=None, 
                        help='Save the quantized model to the specified path!')

    # WandB Arguments
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)


    #Experiments Arguments
    parser.add_argument('--save_name', type=str, default=None, help='The path to save experiment data, '
                                                                    'including quantized models, dumped layer inputs, etc. The data will be saved in experiments/[model]/save_name. Default: [datetime].')
    parser.add_argument('--capture_layer_io', action=argparse.BooleanOptionalAction, default=False,
                        help='Capture the input and output of the specified decoder layer and dump into a file')
    parser.add_argument('--layer_idx', type=int, default=10, help='Which decoder layer to capture')

    # LM Eval Arguments
    parser.add_argument("--lm_eval", action="store_true", help="Evaluate the model on LM Eval tasks.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "boolq"],
    )
    parser.add_argument('--lm_eval_batch_size', type=int, default=32, help='Batch size for evaluating with lm eval harness.')
    parser.add_argument('--distribute', action=argparse.BooleanOptionalAction, default=False,
                        help='Distribute the model on multiple GPUs for evaluatione')

    args = parser.parse_args()
    if args.lm_eval:
        from lm_eval import tasks
        from lm_eval import utils as lm_eval_utils
        from lm_eval.tasks import initialize_tasks
        initialize_tasks()
        for task in args.tasks:
            if task not in lm_eval_utils.MultiChoice(tasks.ALL_TASKS):
                raise ValueError(f"Invalid task: {task}")

    # quant_type = f'w{args.w_bits}a{args.a_bits}_{args.rotate_mode}'
    if args.save_name is None:
        args.save_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    setattr(args, 'save_path',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', args.model, args.save_name))
    os.makedirs(args.save_path, exist_ok=True)

    config_logging(os.path.join(args.save_path, f'{args.save_name}.log'))
    
    # assert args.a_groupsize == args.w_groupsize, 'a_groupsize should be the same as w_groupsize!'
    assert args.k_pre_rope == False, 'Pre-RoPE quantization is not supported yet!'
    if args.w_format == 'nvfp4':
        assert args.w_bits == 4, 'NVFP4 weights require --w_bits 4.'
        if args.w_asym:
            logging.warning('NVFP4 is symmetric; --w_asym is ignored.')
        if args.w_clip:
            logging.warning('NVFP4 RTN does not use --w_clip MSE search.')
        if args.w_groupsize != -1:
            logging.warning('NVFP4 uses group_size=16; --w_groupsize is ignored.')
        args.w_groupsize = -1
    if args.a_format == 'nvfp4':
        assert args.a_bits == 4, 'NVFP4 activations require --a_bits 4.'
        if args.a_asym:
            logging.warning('NVFP4 activations are symmetric; --a_asym is ignored.')
        args.a_groupsize = 16
        args.a_per_tensor = False
    if args.requant:
        assert args.w_bits < 16, 'ReQuant requires quantized weights (w_bits < 16).'
        if args.w_format == 'int':
            assert args.w_groupsize == -1, 'ReQuant INT path only supports per-channel weights (w_groupsize=-1).'
        assert args.requant_sweeps >= 1, 'requant_sweeps T must be >= 1.'
        assert args.requant_neighborhood >= 1, 'requant_neighborhood K must be >= 1.'
    if not args.w_rtn and args.w_format == 'nvfp4' and args.w_bits < 16:
        raise NotImplementedError('NVFP4 currently supports RTN only (--w_rtn).')

    if args.model == 'facebook/opt-125m' or args.model == 'facebook/opt-1.3b':
        logging.warning('Warning: OPT-125M/1.3B is only for debugging purposes!!')


    if args.wandb:
        assert args.wandb_id is not None and args.wandb_project is not None, 'WandB ID/project is not provided!'
        
    logging.info('Arguments: ')
    logging.info(pprint.pformat(vars(args)))
    logging.info('--' * 30)
    return args


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )

def distribute_model(model) -> None:
    """Distribute the model across available GPUs. NB: only implemented for Llama-2."""
    no_split_module_classes = ['LlamaDecoderLayer']
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    print(device_map)
    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

    cleanup_memory()