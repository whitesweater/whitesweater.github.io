def __init__(self, data_name, raw_data, tokenizer, bot, eot):
    super(SupervisedDataset, self).__init__()
    self.tokenizer = tokenizer
    self.data_name = data_name
    questions, cots, answers = [], [], []
    group_keys = []

    # 统计信息
    stats = {
        "total": len(raw_data),
        "processed": 0,
        "skipped_bad_data": 0,
        "skipped_too_long": 0,
        "skipped_invalid_answer": 0,
    }

    print(f"\n{'='*60}")
    print(f"📊 Processing {self.data_name} dataset")
    print(f"{'='*60}")

    # 使用 tqdm 包装 enumerate，显示清晰的进度条
    with tqdm(
            total=len(raw_data),
            desc="🔄 Parsing examples",
            unit="example",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    ) as pbar:
        for num_iter, example in enumerate(raw_data):
            # 实验模式：限制数据量
            if training_args.exp_mode and num_iter >= training_args.exp_data_num:
                pbar.update(len(raw_data) - num_iter)  # 跳过剩余
                break

            q_key = str(example.get("question_id", example["question"])).strip()
            question = f"{example['question']}"

            skip_reason = None

            try:
                if "icot" in self.data_name and "full" in self.data_name:
                    # icot-full (GSM8k-Aug-NL)
                    if example["answer"] is None:
                        skip_reason = "bad_data"
                    else:
                        token_num = len(tokenizer.encode(
                            example["question"] + example["cot"] + example["answer"]
                        ))
                        if token_num > training_args.max_token_num:
                            skip_reason = "too_long"
                        else:
                            cot = f"{example['cot']}".split(". ")
                            if not training_args.include_last_cot:
                                cot = cot[:-1]

                            answer = example["answer"].split(" ")[-1]
                            if not answer[0].isdigit():
                                skip_reason = "invalid_answer"
                            else:
                                answer = f"The answer is: {answer}"
                                answer = answer.replace("####", "")
                                questions.append(question)
                                cots.append(". ".join(cot) + ".\n" if cot else "")
                                answers.append(answer)
                                group_keys.append(q_key)
                                stats["processed"] += 1

                elif "icot" in self.data_name:
                    # icot (GSM8k-Aug)
                    token_num = len(tokenizer.encode(
                        example["question"] + example["cot"] + example["answer"]
                    ))
                    if token_num > training_args.max_token_num:
                        skip_reason = "too_long"
                    else:
                        cot = f"{example['cot']}".split(" ")
                        if not training_args.include_last_cot:
                            cot = cot[:-1]

                        answer = example["answer"].split(" ")[-1]
                        if not answer[0].isdigit():
                            skip_reason = "invalid_answer"
                        else:
                            answer = f"The answer is: {answer}"
                            answer = answer.replace("####", "")
                            questions.append(question)
                            cots.append(" ".join(cot))
                            answers.append(answer)
                            group_keys.append(q_key)
                            stats["processed"] += 1

                elif "commonsense" in self.data_name or "strategy" in self.data_name:
                    question = example["question"].strip() + "\n"
                    cot = example["cot"].strip() + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"

                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num:
                        skip_reason = "too_long"
                    else:
                        questions.append(question)
                        cots.append(cot)
                        answers.append(answer)
                        group_keys.append(q_key)
                        stats["processed"] += 1

                elif "prontoqa" in data_args.data_name:
                    question = example["question"].strip() + "\n"
                    cot = "\n".join(example["steps"][:-1]) + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"

                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num:
                        skip_reason = "too_long"
                    else:
                        questions.append(question)
                        cots.append(cot)
                        answers.append(answer)
                        group_keys.append(q_key)
                        stats["processed"] += 1

                else:
                    raise NotImplementedError

            except Exception as e:
                skip_reason = "error"
                pbar.set_postfix_str(f"⚠️  Error: {str(e)[:30]}")

            # 更新跳过统计
            if skip_reason:
                if skip_reason == "bad_data":
                    stats["skipped_bad_data"] += 1
                elif skip_reason == "too_long":
                    stats["skipped_too_long"] += 1
                elif skip_reason == "invalid_answer":
                    stats["skipped_invalid_answer"] += 1

            # 更新进度条的后缀信息（显示实时统计）
            pbar.set_postfix_str(
                f"✓ {stats['processed']} | "
                f"⏭️  {stats['skipped_bad_data']+stats['skipped_too_long']+stats['skipped_invalid_answer']}"
            )
            pbar.update(1)

    # 打印处理摘要
    print(f"\n{'='*60}")
    print(f"📈 Processing Summary:")
    print(f"{'='*60}")
    print(f"  Total examples:           {stats['total']:>6}")
    print(f"  ✓ Processed:              {stats['processed']:>6} ({stats['processed']/stats['total']*100:.1f}%)")
    print(f"  ⏭️  Skipped (bad data):     {stats['skipped_bad_data']:>6}")
    print(f"  ⏭️  Skipped (too long):     {stats['skipped_too_long']:>6}")
    print(f"  ⏭️  Skipped (invalid ans):  {stats['skipped_invalid_answer']:>6}")
    print(f"{'='*60}\n")

    # 实验模式截断
    if training_args.exp_mode:
        questions = questions[:training_args.exp_data_num]
        cots = cots[:training_args.exp_data_num]
        answers = answers[:training_args.exp_data_num]
        group_keys = group_keys[:training_args.exp_data_num]
        print(f"⚙️  Experiment mode: Limited to {len(questions)} samples\n")

    # 多样本扩展（用于多回答对比学习）
    K = getattr(training_args, "samples_per_group", 1)
    if K > 1:
        print(f"🔄 Expanding samples: {len(questions)} → {len(questions) * K} (K={K} per group)")
        questions = sum([[q] * K for q in questions], [])
        cots = sum([[c] * K for c in cots], [])
        answers = sum([[a] * K for a in answers], [])
        group_keys = sum([[g] * K for g in group_keys], [])

    print(f"📦 Final dataset size: {len(questions)} samples")
    print(f"🏷️  Unique groups: {len(set(group_keys))}\n")

    # Tokenization（调用 preprocess，内部也有进度条）
    self.data_dict = preprocess(questions, cots, answers, tokenizer, bot, eot)
    self.keys = list(self.data_dict.keys())

    # 生成 group_ids
    self.group_ids = torch.tensor(
        [_stable_hash(k) for k in group_keys], dtype=torch.long
    )
    self.keys.append("group_ids")
    self.data_dict["group_ids"] = self.group_ids

    print(f"✅ Dataset ready: {len(self)} samples, {len(set(self.group_ids.tolist()))} groups\n")
