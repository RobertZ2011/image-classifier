from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import PIL
import argparse
from datasets import concatenate_datasets, Image, load_dataset
from tqdm.auto import tqdm

def load_data(name, seed=42):
    return load_dataset(name, num_proc=8).cast_column('image', Image(mode='RGB')).shuffle(seed=seed)

def partition_data(data, t, split=-1):
    if split == -1:
        return data[t]
    else:
        combined = concatenate_datasets([data['test'], data['train']])
        total = len(combined)

        if t == 'train':
             return combined.select(range(split))
        elif t == 'test':
            return combined.select(range(split, total))

        raise ValueError('Unsupported data set type')

def process_data(data, processor):
    def transform(example_batch):
        inputs = processor(images=example_batch['image'], return_tensors='pt')
        example_batch['pixel_values'] = inputs['pixel_values']
        return example_batch

    return data.map(transform, batched=True, num_proc=8).with_format('torch', columns=['pixel_values'], output_all_columns=True)

def process_training_data(data, processor, split=-1):
    data = partition_data(data, 'train', split=split)
    return process_data(data, processor)

def process_test_data(data, processor, split=-1):
    data = partition_data(data, 'test', split=split)
    return process_data(data, processor)

def create_pipeline(model, processor, device):
   return pipeline('image-classification', device=device, model=model, image_processor=processor)

def run(pipe, images):
    results = pipe(images)
    for predictions in results:
        print('')
        for prediction in predictions:
            print(f"{prediction['label']} {prediction['score']:.03f}")

def finetune(model, target_device, data, out_dir, num_epochs=12):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True
    
    dataloader = DataLoader(data, shuffle=True, batch_size=8)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        last_loss = 0
        last_lr = 0

        for batch in dataloader:
            batch = { k: v.to(target_device) for k, v in batch.items() }
            outputs = model(pixel_values=batch['pixel_values'], labels=batch['label'])

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            last_loss = loss.item()
            last_lr = optimizer.param_groups[0]['lr']

            progress_bar.update(1)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, loss: {last_loss}, lr: {last_lr}')

        
    
    model.save_pretrained(save_directory=out_dir)

def evaluate(pipe, dataset):
    top_1 = 0
    top_5 = 0
    total = len(dataset)
    out = pipe(dataset['image'])
    for (p, t) in zip(out, dataset['label']):
        for i in range(0, 5):
            label = p[i]['label']
            if t == pipe.model.config.label2id[label]:
                if i == 0:
                    top_1 += True
                
                top_5 += 1

    print(f'Top1: {100 * float(top_1) / total}% ({top_1}/{total})')
    print(f'Top5: {100 * float(top_5) / total}% ({top_5}/{total})')

def main():
    parser = argparse.ArgumentParser(description='Image Classifier')
    subparsers = parser.add_subparsers(dest='command', help='Sub-commands (tune, eval, run)')
    subparsers.required = True

    # tune subcommand
    parser_tune = subparsers.add_parser('tune', help='tune the model')
    parser_tune.add_argument('-e', '--epochs', type=int, help='Number of epochs')
    parser_tune.add_argument('-m', '--model', type=str, help='Model name/path')
    parser_tune.add_argument('-d', '--dataset', type=str, help='Dataset for evaluation')
    parser_tune.add_argument('-s', '--split', type=int, default=-1)
    parser_tune.add_argument('-p', '--processor', type=str, default='auto', help='Device to use')
    parser_tune.add_argument('out_dir', type=str, help='Output directory')

    # Eval subcommand
    parser_eval = subparsers.add_parser('eval', help='Evaluate the model')
    parser_eval.add_argument('-m', '--model', type=str, help='Model name/path')
    parser_eval.add_argument('-d', '--dataset', type=str, help='Dataset for evaluation')
    parser_eval.add_argument('-s', '--split', type=int, default=-1)
    parser_eval.add_argument('-p', '--processor', type=str, default='auto', help='Device to use')

    # Run subcommand
    parser_run = subparsers.add_parser('run', help='Run the model')
    parser_run.add_argument('-m', '--model', type=str, help='Model name/path')
    parser_run.add_argument('-p', '--processor', type=str, default='auto', help='Device to use')
    parser_run.add_argument('images', nargs='+', type=str, help='Path to the images to classify')

    args = parser.parse_args()

    target_device = torch.device(args.processor)
    processor = AutoImageProcessor.from_pretrained(args.model)

    if args.command == 'tune':
        data = load_data(args.dataset)
        model = AutoModelForImageClassification.from_pretrained(
            args.model,
            num_labels=len(data['train'].features['label'].names),
            ignore_mismatched_sizes=True
        ).to(target_device)
        training = process_training_data(data, processor, split=args.split)
        training_filtered = training.remove_columns(['bbox', 'image'])
        finetune(model, target_device, training_filtered, args.out_dir)

        test = process_test_data(data, processor, split=args.split)
        training_eval_len = min(len(test), len(training))
        print('Test dataset:')
        evaluate(create_pipeline(model, processor, target_device), test)
        print('Training dataset:')
        evaluate(create_pipeline(model, processor, target_device),
            training.select(range(min(len(test), len(training))))
        )

    elif args.command == 'eval':
        model = AutoModelForImageClassification.from_pretrained(args.model).to(target_device)
        data = load_data(args.dataset)
        test = process_test_data(data, processor, split=args.split)
        training = process_training_data(data, processor, split=args.split)

        print('Test dataset:')
        evaluate(create_pipeline(model, processor, target_device), test)
        print(f'Training dataset:')
        evaluate(create_pipeline(model, processor, target_device),
            training.select(range(min(len(test), len(training))))
        )
    elif args.command == 'run':
        model = AutoModelForImageClassification.from_pretrained(args.model).to(target_device)
        images = list(map(lambda x: PIL.Image.open(x).convert('RGB'), args.images))
        run(create_pipeline(model, processor, target_device), images)

if __name__ == '__main__':
    main()