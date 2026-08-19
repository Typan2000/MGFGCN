import torch
import numpy as np
import argparse
import time
import util
import logging
import os
import json
from engine import trainer
from pathlib import Path
from datetime import datetime

def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)

def load_dataset_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def build_logger():
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    Path("log/").mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fh = logging.FileHandler(f"log/{current_time}_MGFGCN.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def format_iter_log(epoch, iteration, loss, mape, rmse):
    return (
        f"[Epoch {epoch:03d} | Iter {iteration:03d}] "
        f"train_loss={loss:.4f} | train_mape={mape:.4f} | train_rmse={rmse:.4f}"
    )

def format_epoch_log(epoch, train_metrics, valid_metrics, train_secs, infer_secs, best_loss, patience_count, checkpoint_path, is_best):
    status_text = (
        f"Status | best_valid_loss={best_loss:.4f} | best" '\n'
        f"Checkpoint saved: " + checkpoint_path
        if is_best else
        f"Status | best_valid_loss={best_loss:.4f} | no improvement for {patience_count} epoch"
    )
    return '\n'.join([
        '=' * 88,
        f"Epoch {epoch:03d}",
        f"Train | loss={train_metrics[0]:.4f} | mape={train_metrics[1]:.4f} | rmse={train_metrics[2]:.4f} | time={train_secs:.2f}s",
        f"Valid | loss={valid_metrics[0]:.4f} | mape={valid_metrics[1]:.4f} | rmse={valid_metrics[2]:.4f} | time={infer_secs:.2f}s",
        status_text,
        '=' * 117
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/PeMS08.json', help='config_file')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu')
    args = parser.parse_args()
    config_path = args.config_path
    config = load_dataset_config(config_path)
    logger = build_logger()
    logger.info("Loaded config: %s", config)
    if not os.path.exists(config['save']):
        os.makedirs(config['save'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device(args.device)
    dataloader = util.load_dataset(config['data'], config['N_t'], config['batch_size'], config['batch_size'], config['batch_size'])
    distance_matrix = torch.load(config['distance_matrix'])
    scaler = dataloader['scaler']
    engine = trainer(config, scaler, distance_matrix, args.device)

    logger.info("%s", '=' * 88)
    logger.info("Start training")
    his_loss = []
    val_time = []
    train_time = []
    count = 0
    mtrain_loss_list = []
    mval_loss_list = []

    for i in range(1, config['max_epoch'] + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y, t_i) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            metrics = engine.train(trainx, trainy[:, 0, :, :], t_i)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % config['print_every'] == 0:
                logger.info(
                    format_iter_log(i, iter, train_loss[-1], train_mape[-1], train_rmse[-1])
                )
        engine.scheduler.step()
        t2 = time.time()
        train_time.append(t2 - t1)
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y, t_i) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testy = torch.Tensor(y).to(device)
            metrics = engine.eval(testx, testy[:, 0, :, :], t_i)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mtrain_loss_list.append(mtrain_loss)
        mval_loss_list.append(mvalid_loss)

        is_best = len(his_loss) == 0 or mvalid_loss < np.min(his_loss)
        if is_best:
            count = 0
            checkpoint_path = config['save'] + 'MGFGCN_' + config['dataset_class'] + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth"
            torch.save(engine.model.state_dict(), checkpoint_path)
        else:
            count += 1
        his_loss.append(mvalid_loss)
        best_valid_loss = np.min(his_loss)
        logger.info(
            format_epoch_log(i,(mtrain_loss, mtrain_mape, mtrain_rmse),(mvalid_loss, mvalid_mape, mvalid_rmse),
                t2 - t1, s2 - s1,
                best_valid_loss,
                count,
                checkpoint_path,
                is_best
            )
        )
        if count >= config['patience']:
            logger.info("Early stopping triggered at epoch %03d", i)
            break
    logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(
        config['save'] + 'MGFGCN_' + config['dataset_class'] + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy[:, 0, :, :]

    for iter, (x, y, t_i) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        with torch.no_grad():
            preds = engine.model(testx, t_i)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...].transpose(1, 2)
    logger.info("Training finished")
    log = 'The valid loss on best model is {:.4f}'
    logger.info(log.format(his_loss[bestid]))

    amae = []
    amape = []
    armse = []
    for i in range(config['output_dim']):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        logger.info(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over '+ str(config['output_dim']) + ' horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    logger.info(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    logger.info(config)

if __name__ == "__main__":
    main()

