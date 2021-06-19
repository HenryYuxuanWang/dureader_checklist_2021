from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./outputs',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    max_steps=0,
    learning_rate=3e-5,
    gradient_accumulation_steps=1,
    warmup_ratio=0.1,
    weight_decay=0.01,
    adam_epsilon=1e-8,
    seed=42,
    max_grad_norm=1.0,
    save_steps=1000
)

training_args.max_seq_length = 512
training_args.data_dir = './dataset'
training_args.doc_stride = 128
training_args.max_query_length = 64
training_args.max_answer_length = 512
training_args.do_lower_case = True
training_args.verbose = False
training_args.version_2_with_negative = True
training_args.null_score_diff_threshold = 0
training_args.n_best_size = 20
training_args.logging_steps = 50
training_args.cls_threshold = 0.7
training_args.tokenizer_dir = './chinese-roberta-wwm-ext'

