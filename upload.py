import argparse
import wandb

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--project', 
                    help='project name', default='test')
parser.add_argument('--name', 
                    help='exp name', default='test')
parser.add_argument('--log_file', 
                    help='file path for the logs)', default='outputs_requeue_v2/imgnt1k/decoupled_0.20_cache/ae31717c-65a8-4f1b-82ff-4cef059fb916/log')
    
import json

args = parser.parse_args()


# load configs
f = open(args.log_file.replace('log', 'params.json'))
config = json.load(f)
print("config:")
print(json.dumps(config, indent=2))
wandb.init(project=args.project, entity="pchiang", name=args.name, 
                config=config)

# uploading log files
f = open(args.log_file)
latest_epoch = -1
top_1_best = -1
for line in f:
    data= json.loads(line)
    epoch = data['epoch']
    if epoch > latest_epoch:
        if 'top_1_best' not in data:
            top_1_best = max(data['top_1'], top_1_best)
            data['top_1_best'] = top_1_best
        print(f"uploading logs for epoch {epoch}")

        latest_epoch = epoch
        wandb.log(data)
f.close()




# log the files