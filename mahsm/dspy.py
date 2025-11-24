"""
mahsm.dspy - Complete DSPy re-export

This module re-exports the entire DSPy library for convenience.
Use `ma.dspy.*` to access any DSPy functionality.
"""

from dspy import Adapter, Audio, AvatarOptimizer, BaseLM, BaseModule, BestOfN, BetterTogether, BootstrapFewShot, BootstrapFewShotWithOptuna, BootstrapFewShotWithRandomSearch, BootstrapFinetune, BootstrapRS, COPRO, ChainOfThought, ChatAdapter, Code, CodeAct, ColBERTv2, Completions, DSPY_CACHE, Embedder, Embeddings, Ensemble, Evaluate, Example, GEPA, History, Image, InferRules, InputField, JSONAdapter, KNN, KNNFewShot, LM, LabeledFewShot, MIPROv2, Module, MultiChainComparison, OldField, OldInputField, OldOutputField, OutputField, Parallel, Predict, Prediction, ProgramOfThought, Provider, PythonInterpreter, ReAct, Refine, Retrieve, SIMBA, Signature, SignatureMeta, Tool, ToolCalls, TrainingJob, TwoStepAdapter, Type, XMLAdapter, asyncify, bootstrap_trace_data, cache, configure, configure_cache, configure_dspy_loggers, context, disable_litellm_logging, disable_logging, enable_litellm_logging, enable_logging, ensure_signature, infer_prefix, inspect_history, load, majority, make_signature, settings, streamify, syncify, track_usage
from dspy.adapters.baml_adapter import Any, BAMLAdapter, BaseModel, COMMENT_SYMBOL, Literal, Union, get_args, get_origin, original_format_field_value
from dspy.adapters.base import BaseCallback, TYPE_CHECKING, logger, split_message_content_for_custom_types, with_callbacks
from dspy.adapters.chat_adapter import AdapterParseError, ContextWindowExceededError, FieldInfo, FieldInfoWithName, NamedTuple, field_header_pattern, format_field_value, get_annotation_name, get_field_description_string, parse_value, translate_field_type
from dspy.adapters.json_adapter import serialize_for_json
from dspy.adapters.types.audio import SF_AVAILABLE, encode_audio
from dspy.adapters.types.base_type import CUSTOM_TYPE_END_IDENTIFIER, CUSTOM_TYPE_START_IDENTIFIER
from dspy.adapters.types.code import ClassVar, create_model
from dspy.adapters.types.image import PIL_AVAILABLE, encode_image, is_image, is_url, urlparse
from dspy.adapters.types.tool import Callable, TypeAdapter, ValidationError, convert_input_schema_to_tool_args, get_type_hints, validate
from dspy.adapters.utils import Mapping, find_enum_member, get_dspy_field_type
from dspy.clients import Cache, DISK_CACHE_DIR, DISK_CACHE_LIMIT, Path, configure_litellm_logging
from dspy.clients.base_lm import GLOBAL_HISTORY, MAX_HISTORY_SIZE, pretty_print_history
from dspy.clients.cache import FanoutCache, LRUCache, request_cache, sha256, wraps
from dspy.clients.databricks import DatabricksProvider, TrainDataFormat, TrainingJobDatabricks, get_finetune_directory
from dspy.clients.lm import MemoryObjectSendStream, OpenAIProvider, ReinforceJob, alitellm_completion, alitellm_responses_completion, alitellm_text_completion, cast, litellm_completion, litellm_responses_completion, litellm_text_completion
from dspy.clients.lm_local import LocalProvider, create_output_dir, encode_sft_example, get_free_port, save_data, train_sft_locally, wait_for_server
from dspy.clients.lm_local_arbor import ArborProvider, ArborReinforceJob, ArborTrainingJob, GRPOGroup, GRPOTrainKwargs, TrainingStatus, TypedDict, datetime
from dspy.clients.openai import TrainingJobOpenAI
from dspy.clients.provider import Future, Thread, abstractmethod
from dspy.clients.utils_finetune import DSPY_CACHEDIR, Enum, GRPOChatData, Message, MessageAssistant, find_data_error_chat, find_data_error_chat_message, find_data_errors_completion, infer_data_format, validate_data_format, write_lines
from dspy.datasets import AlfWorld, Colors, DataLoader, Dataset, HotPotQA, MATH
from dspy.datasets.alfworld.alfworld import EnvPool, env_worker
from dspy.datasets.colors import all_colors
from dspy.datasets.dataset import dotdict
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric, parse_integer_answer
from dspy.datasets.math import extract_answer
from dspy.dsp.colbertv2 import ColBERTv2RerankerLocal, ColBERTv2RetrieverLocal, colbertv2_get_request, colbertv2_get_request_v2, colbertv2_get_request_v2_wrapped, colbertv2_post_request, colbertv2_post_request_v2, colbertv2_post_request_v2_wrapped
from dspy.dsp.utils import DEFAULT_CONFIG, DPR_normalize, DPR_tokenize, NullContextManager, STokenizer, Settings, SimpleTokenizer, Tokenizer, Tokens, batch, config_owner_async_task, config_owner_thread_id, contextmanager, create_directory, deduplicate, defaultdict, dotdict_lax, file_tqdm, flatten, global_lock, groupby_first_item, grouper, has_answer, int_or_float, lengths2offsets, load_batch_backgrounds, locate_answers, main_thread_config, print_message, process_grouped_by_first_item, strip_accents, thread_local_overrides, timestamp, zip_first, zipstar
from dspy.evaluate import CompleteAndGrounded, EM, SemanticF1, answer_exact_match, answer_passage_match, normalize_text
from dspy.evaluate.auto_evaluation import AnswerCompleteness, AnswerGroundedness, DecompositionalSemanticRecallPrecision, SemanticRecallPrecision, f1_score
from dspy.evaluate.evaluate import EvaluationResult, HTML, ParallelExecutor, configure_dataframe_for_ipython_notebook_display, display, display_dataframe, is_in_ipython_notebook_environment, merge_dicts, prediction_is_dictlike, stylize_metric_name, truncate_cell
from dspy.evaluate.metrics import Counter, F1, HotPotF1, em_score, hotpot_f1_score, precision_score
from dspy.predict.aggregation import default_normalize
from dspy.predict.avatar import Action, ActionOutput, Actor, Avatar, Field, deepcopy, get_number_with_suffix
from dspy.predict.parameter import Parameter
from dspy.predict.predict import serialize_object
from dspy.predict.refine import OfferFeedback, inspect_modules, recursive_mask
from dspy.primitives.base_module import Generator, deque, get_dependency_versions
from dspy.primitives.module import ProgramMeta, set_attribute_by_name
from dspy.primitives.python_interpreter import InterpreterError, PathLike, TracebackType
from dspy.propose import GroundedProposer
from dspy.propose.dataset_summary_generator import DatasetDescriptor, DatasetDescriptorWithPriorObservations, ObservationSummarizer, create_dataset_summary, order_input_keys_in_string, strip_prefix
from dspy.propose.grounded_proposer import DescribeModule, DescribeProgram, GenerateModuleInstruction, MAX_INSTRUCT_IN_HISTORY, Proposer, TIPS, create_example_string, create_predictor_level_history_string, generate_instruction_class, get_dspy_source_code, get_prompt_model, get_signature
from dspy.propose.propose_base import ABC
from dspy.propose.utils import create_instruction_set_history_string, extract_symbols, get_program_instruction_set_string, new_getfile, parse_list_of_instructions       
from dspy.retrievers.embeddings import Unbatchify
from dspy.retrievers.retrieve import single_query_passage
from dspy.signatures.field import DSPY_FIELD_ARG_NAMES, PYDANTIC_CONSTRAINT_MAP, move_kwargs, new_to_old_field
from dspy.streaming import StatusMessage, StatusMessageProvider, StreamListener, StreamResponse, apply_sync_streaming, streaming_response
from dspy.streaming.messages import StatusStreamingCallback, dataclass, sync_send_to_stream
from dspy.streaming.streamify import AsyncGenerator, Awaitable, ModelResponseStream, Queue, create_memory_object_stream, create_task_group, find_predictor_for_stream_listeners, iscoroutinefunction
from dspy.streaming.streaming_listener import ADAPTER_SUPPORT_STREAMING
from dspy.teleprompt import Teleprompter
from dspy.teleprompt.avatar_optimizer import Comparator, DEFAULT_MAX_EXAMPLES, EvalResult, FeedbackBasedInstruction, ThreadPoolExecutor, sample, tqdm
from dspy.teleprompt.bettertogether import all_predictors_have_lms, kill_lms, launch_lms, prepare_student
from dspy.teleprompt.bootstrap_finetune import FinetuneTeleprompter, assert_no_shared_predictor, assert_structural_equivalency, build_call_data_from_trace, copy_program_with_lms, get_unique_lms, prepare_teacher
from dspy.teleprompt.bootstrap_trace import FailedPrediction, MethodType, TraceData
from dspy.teleprompt.copro_optimizer import BasicGenerateInstruction, GenerateInstructionGivenAttempts
from dspy.teleprompt.gepa.gepa import AUTO_RUN_SETTINGS, DspyGEPAResult, GEPAFeedbackMetric, Optional, Protocol
from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, DspyAdapter, EvaluationBatch, GEPAAdapter, LoggerAdapter, PredictorFeedbackFn, ScoreWithFeedback
from dspy.teleprompt.grpo import GRPO, disable_lm_cache, recover_lm_cache
from dspy.teleprompt.infer_rules import RulesInductionProgram
from dspy.teleprompt.mipro_optimizer_v2 import BLUE, BOLD, BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT, ENDC, GREEN, LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT, MIN_MINIBATCH_SIZE, YELLOW, create_minibatch, create_n_fewshot_demo_sets, eval_candidate_program, get_program_with_highest_avg_score, print_full_program, save_candidate_program, set_signature
from dspy.teleprompt.signature_opt import SignatureOptimizer
from dspy.teleprompt.simba import append_a_demo, append_a_rule, prepare_models_for_resampling, wrap_program
from dspy.teleprompt.utils import calculate_last_n_proposed_quality, eval_candidate_program_with_pruning, get_task_model_history_for_full_example, get_token_usage, log_token_usage, old_getfile, save_file_to_log_dir, setup_logging
from dspy.utils import DummyLM, DummyVectorizer, download, dummy_rm
from dspy.utils.asyncify import CapacityLimiter, get_async_max_workers, get_limiter
from dspy.utils.caching import create_subdir_in_cachedir
from dspy.utils.callback import ACTIVE_CALL_ID, ContextVar
from dspy.utils.hasher import Hasher, dumps
from dspy.utils.langchain_tool import convert_langchain_tool
from dspy.utils.logging_utils import DSPY_LOGGING_STREAM, DSPyLoggingStream, LOGGING_DATETIME_FORMAT, LOGGING_LINE_FORMAT
from dspy.utils.mcp import convert_mcp_tool
from dspy.utils.parallelizer import FIRST_COMPLETED, wait
from dspy.utils.syncify import run_async
from dspy.utils.usage_tracker import UsageTracker