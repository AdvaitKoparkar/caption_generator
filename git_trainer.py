import os
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from datasets import load_dataset

class GiTFineTuningConfig:
    model_name = "microsoft/git-base"
    seed = 2025
    num_samples = 128
    train_split = 0.8
    validate_every = 15
    batch_size = 32
    optimizer_config = {
        'lr': 1e-5,
    }
    save_path = ''

class ImageCaptioningDataset(torch.utils.data.Dataset):
    def __init__(self, data : torch.utils.data.Dataset, processor : AutoProcessor ):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.processor(images=item["image"], text=item["caption"], padding="max_length", return_tensors="pt")
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")

class DDPTrainer:
    def __init__(self, model : torch.nn.Module, 
                 dataloader_train : torch.utils.data.DataLoader, 
                 dataloader_val : torch.utils.data.DataLoader, 
                 optimizer : torch.optim.Optimizer):
        self.device_id = int(os.environ['LOCAL_RANK'])
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.epochs_run = 0
        self.best_val_loss = float('inf')
        self.model = DDP(model.to(self.device_id), device_ids=[self.device_id])
        if os.path.exists(GiTFineTuningConfig.save_path):
            print("Loading snapshot")
            self._load_snapshot(GiTFineTuningConfig.save_path)

    def _step(self, source : dict , targets: torch.Tensor ):
        self.optimizer.zero_grad()
        outputs = self.model(input_ids=source['input_ids'], pixel_values=source['pixel_values'], labels=targets)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch : int ):
        b_sz = len(next(iter(self.dataloader_train))[0])
        self.dataloader_train.sampler.set_epoch(epoch)
        pbar = tqdm(self.dataloader_train)
        pbar.set_description(f"[GPU{self.device_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.dataloader_train)}")
        self.model.train()
        for source, targets in pbar:
            source = source.to(self.device_id)
            targets = targets.to(self.device_id)
            self._step(source, targets)

    def _save_snapshot(self, epoch):
        val_loss = 0.0
        print('Running val')
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(self.dataloader_val)
            for source, targets in pbar:
                source = source.to(self.device_id)
                targets = targets.to(self.device_id)
                output = self.model(source)
                l = self.loss(output, targets).item()
                val_loss += l

            val_loss /= len(self.dataloader_val)
            print(f'Val loss: {val_loss:.04f}, Best before this: {self.best_val_loss:04f}')

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            snapshot = {
                "MODEL_STATE": self.model.module.state_dict(),
                "OPTIMIZER_STATE": self.optimizer.state_dict(),
                "VAL_LOSS": self.best_val_loss,
                "EPOCHS_RUN": epoch,
            }
            torch.save(snapshot, self.config['save_path'])
            print(f"Epoch {epoch} | Training snapshot saved at {self.config['save_path']}")

    def _load_snapshot(self, snapshot_path : str ):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.best_val_loss = snapshot["BEST_VAL_LOSS"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}, Val loss: {self.best_val_loss:.04f}")

    def train(self, ):
        for epoch in range(self.epochs_run, self.config['num_epochs']):
            self._run_epoch(epoch)
            if self.device_id == 0 and epoch % self.config['save_every'] == 0:
                self._save_snapshot(epoch)

def load_training_artifact():
    data = load_dataset("mrSoul7766/instagram_post_captions", split=f'train[0:{GiTFineTuningConfig.num_samples}]')
    processor = AutoProcessor.from_pretrained(GiTFineTuningConfig.model_name)
    dataset = ImageCaptioningDataset(data, processor)
    data_train, data_val = torch.utils.data.random_split(dataset, [GiTFineTuningConfig.train_split, 1-GiTFineTuningConfig.train_split])
    model = AutoModelForCausalLM.from_pretrained(GiTFineTuningConfig.model_name)
    opt = torch.optim.Adam(model.parameters(), **GiTFineTuningConfig.optimizer_config)
    return model, data_train, data_val, opt

def get_dataloader(dataset : torch.utils.data.Dataset ) :
    dl = torch.utils.data.DataLoader(
        dataset, 
        batch_size=GiTFineTuningConfig.batch_size,
        pin_memory=True,
        shuffle=False, # handled by distributed sampler
        sampler=torch.utils.data.distributed.DistributedSampler(dataset),
    )
    return dl

def train():
    model, data_train, data_val, opt = load_training_artifact()
    ddp_setup()
    model, data_train, data_val, opt, loss = load_training_artifact()
    dataloader_train = get_dataloader(data_train)
    dataloader_val = get_dataloader(data_val)
    opt = torch.optim.AdamW(model.parameters(), **GiTFineTuningConfig.optimizer_config)
    trainer = DDPTrainer(model, dataloader_train, dataloader_val, opt)
    trainer.train()

if __name__ == '__main__':
    
    import pdb; pdb.set_trace()