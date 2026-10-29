# FPUnet weights

The `.pth` files are git-ignored. Download them from
[Google Drive](https://drive.google.com/drive/folders/1jXtpTeI6MsttXhTSDTYItPNMMTF7igY6?usp=drive_link)
into this directory:

- `best_fpunet_256_8.pth`: used by `config/map_prediction.yaml` and `config/online_update.yaml`
- `best_model.pth`

```sh
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1jXtpTeI6MsttXhTSDTYItPNMMTF7igY6 -O .
```
