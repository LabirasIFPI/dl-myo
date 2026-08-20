import asyncio
import multiprocessing
import queue
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from myo import MyoClient
from myo.types import EMGMode, IMUMode, ClassifierMode

# ==========================================
# 1. SETUP DO DL-MYO (BLUETOOTH ASSÍNCRONO)
# ==========================================
class PlotClient(MyoClient):
    # Remova o __init__ que estava aqui!

    async def on_emg_data(self, emg):
        # A Myo envia 2 amostras por pacote. Vamos colocar as duas na fila.
        self.q.put(emg.sample1)
        self.q.put(emg.sample2)

    # --- Métodos abstratos obrigatórios ---
    async def on_aggregated_data(self, ad): pass
    async def on_classifier_event(self, ce): pass
    async def on_motion_event(self, me): pass
    async def on_emg_data_aggregated(self, eds): pass
    async def on_fv_data(self, fvd): pass
    async def on_imu_data(self, imu): pass

async def run_myo_bluetooth(q, mac_address):
    logging.info("Conectando à Myo via Bluetooth...")
    client = await PlotClient.with_device(mac=mac_address)
    client.q = q
    
    # Habilitamos o envio de EMG e desligamos o resto
    await client.setup(
        classifier_mode=ClassifierMode.DISABLED,
        emg_mode=EMGMode.SEND_EMG, 
        imu_mode=IMUMode.NONE
    )
    
    await client.start()
    logging.info("Começando a enviar dados para o gráfico!")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await client.stop()
        await client.disconnect()

def worker(q, mac_address):
    """Função que roda em um processo separado para gerenciar o Bluetooth"""
    try:
        asyncio.run(run_myo_bluetooth(q, mac_address))
    except Exception as e:
        print(f"\n[ERRO BLUETOOTH] Falha na conexão: {e}")
        print("Dica: Verifique se a pulseira está ligada, recarregada e próxima ao computador.")


# ==========================================
# 2. SETUP DO PLOT (MATPLOTLIB)
# ==========================================
if __name__ == '__main__':
    MAC = "E8:59:E3:95:37:F1"  # Seu MAC Address

    # Fila de comunicação entre os processos
    q = multiprocessing.Queue()

    # Inicia o processo do Bluetooth em segundo plano
    p = multiprocessing.Process(target=worker, args=(q, MAC))
    p.start()

    print("Aguardando conexão com a pulseira...")
    while q.empty():
        if not p.is_alive():
            print("\nEncerrando: O processo Bluetooth falhou antes de enviar dados.")
            quit()

    # Configuração do Gráfico (baseado no pyomyo)
    QUEUE_SIZE = 100
    SENSORS = 8
    plt.rcParams["figure.figsize"] = (6, 8)
    fig, subplots = plt.subplots(SENSORS, 1)
    fig.canvas.manager.set_window_title("8 Channel EMG plot - DL-Myo")
    fig.tight_layout()

    lines = []
    cmap = plt.get_cmap("tab10")
    colors = cmap.colors

    # Inicializa as linhas dos 8 gráficos
    for i in range(0, SENSORS):
        ch_line, = subplots[i].plot(range(QUEUE_SIZE), [0]*QUEUE_SIZE, color=colors[i])
        lines.append(ch_line)
        subplots[i].set_ylim(-128, 127) # Limites padrão do sinal RAW EMG da Myo

    emg_queue = queue.Queue(QUEUE_SIZE)

    # Função de animação que roda a cada frame
    def animate(frame):
        # Esvazia os dados recentes do bluetooth para a fila local
        while not q.empty():
            myox = list(q.get())
            if emg_queue.full():
                emg_queue.get() # Remove o dado mais antigo
            emg_queue.put(myox)

        if emg_queue.full():
            channels = np.array(emg_queue.queue)
            for i in range(0, SENSORS):
                channel = channels[:, i]
                lines[i].set_ydata(channel)

    # Inicia a animação da janela a cada 20ms
    anim = animation.FuncAnimation(fig, animate, blit=False, interval=20, cache_frame_data=False)

    # Para garantir que o processo bluetooth feche ao fechar a janela
    def on_close(event):
        p.terminate()
        print("Janela fechada, encerrando processos...")

    fig.canvas.mpl_connect('close_event', on_close)

    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close()
        p.terminate()
        quit()