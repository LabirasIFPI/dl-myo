import argparse
import asyncio
import logging
import csv
import time
import os
from datetime import datetime

from myo import MyoClient
from myo.types import ClassifierMode, EMGMode, IMUMode

class RecordClient(MyoClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Cria a pasta datasets se não existir
        os.makedirs("datasets", exist_ok=True)
        
        # Cria o nome do arquivo com timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"datasets/myo_dataset_{timestamp_str}.csv"
        
        # Abre o arquivo CSV
        self.csv_file = open(self.filename, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            "timestamp", "type", 
            "v1", "v2", "v3", "v4", 
            "v5", "v6", "v7", "v8", 
            "v9", "v10"
        ])
        self.start_time = time.time()

    async def on_emg_data(self, emg):
        ts = time.time() - self.start_time
        # A pulseira envia duas amostras de EMG (cada com 8 canais) por pacote
        self.writer.writerow([ts, "EMG"] + list(emg.sample1) + ["", ""])
        self.writer.writerow([ts, "EMG"] + list(emg.sample2) + ["", ""])

    async def on_imu_data(self, imu):
        ts = time.time() - self.start_time
        # v1..v4 = orientation (quaternion)
        # v5..v7 = accelerometer (x, y, z)
        # v8..v10 = gyroscope (x, y, z)
        row = [
            ts, "IMU",
            imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z,
            imu.accelerometer[0], imu.accelerometer[1], imu.accelerometer[2],
            imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]
        ]
        self.writer.writerow(row)

    # --- Funções vazias para ignorar outros tipos de dados ---
    async def on_aggregated_data(self, ad): pass
    async def on_classifier_event(self, ce): pass
    async def on_motion_event(self, me): pass
    async def on_emg_data_aggregated(self, eds): pass
    async def on_fv_data(self, fvd): pass

    def cleanup(self):
        self.csv_file.close()

async def main(args: argparse.Namespace):
    logging.info("Procurando pulseira Myo...")
    # Tenta conectar. Se não houver MAC, conecta na primeira que encontrar.
    client = await RecordClient.with_device(mac=args.mac)

    # Habilita dados brutos de EMG e do IMU
    await client.setup(
        classifier_mode=ClassifierMode.DISABLED,
        emg_mode=EMGMode.SEND_EMG,
        imu_mode=IMUMode.SEND_DATA,
    )
    
    await client.start()
    
    logging.info(f"==== GRAVANDO DADOS POR {args.seconds} SEGUNDOS ====")
    logging.info("=> Faça os movimentos que deseja gravar!")
    
    # Aguarda o tempo especificado capturando eventos
    await asyncio.sleep(args.seconds)
    
    logging.info("==== GRAVAÇÃO CONCLUÍDA ====")
    logging.info(f"Salvado em: {client.filename}")
    
    await client.stop()
    client.cleanup()
    await client.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gravar dados brutos (EMG e IMU) da Myo em um arquivo CSV")
    parser.add_argument("--mac", default="", help="MAC address da pulseira (se vazio conecta na primeira)")
    parser.add_argument("--seconds", default=10, type=int, help="Segundos para gravar os dados (padrão: 10s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main(args))
