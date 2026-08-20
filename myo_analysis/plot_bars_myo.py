import asyncio
import multiprocessing
import queue
import logging
import matplotlib.pyplot as plt
from matplotlib import animation

from myo import MyoClient
from myo.types import EMGMode, IMUMode, ClassifierMode

# ==========================================
# 1. SETUP DO DL-MYO (BLUETOOTH ASSÍNCRONO)
# ==========================================
class PlotClient(MyoClient):
    async def on_emg_data(self, emg):
        # Envia as amostras mais recentes para a fila do gráfico
        self.q.put(emg.sample2)

    # Ignora outros tipos de dados
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
    
    await client.setup(
        classifier_mode=ClassifierMode.DISABLED,
        emg_mode=EMGMode.SEND_EMG, 
        imu_mode=IMUMode.NONE
    )
    
    await client.start()
    logging.info("Começando a enviar dados para o gráfico de barras!")
    
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
# 2. SETUP DO PLOT HEATMAP/BARS (MATPLOTLIB)
# ==========================================
if __name__ == '__main__':
    # Pode passar "" para conectar na primeira pulseira encontrada
    MAC = "E8:59:E3:95:37:F1" 

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(q, MAC))
    p.start()

    print("Aguardando conexão com a pulseira...")
    while q.empty():
        if not p.is_alive():
            print("\nEncerrando: O processo Bluetooth falhou antes de enviar dados.")
            quit()

    SENSORS = 8
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.style.use('dark_background')  # Fica mais bonito para mapa de calor
    
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Myo Heatmap/Bars - Intensidade")
    
    # Mapeia a intensidade para cores (mapa Red/Laranja)
    cmap = plt.get_cmap("inferno")
    
    # Inicia as barras com valor 0
    bars = ax.bar(range(1, SENSORS + 1), [0]*SENSORS, color=cmap(0.0))
    
    ax.set_ylim(0, 128)
    ax.set_xticks(range(1, SENSORS + 1))
    ax.set_xlabel("Canal (Sensor da Pulseira)")
    ax.set_ylabel("Amplitude Muscular Relativa")
    ax.set_title("Intensidade EMG em Tempo Real")

    def animate(frame):
        latest_data = None
        # Pega sempre o último pacote mais fresco disponível
        while not q.empty():
            latest_data = list(q.get())

        if latest_data:
            # Transformamos em valor absoluto para ter noção de força/intensidade
            intensities = [abs(val) for val in latest_data]
            
            for bar, intensity in zip(bars, intensities):
                bar.set_height(intensity)
                # Altera a cor de acordo com a força (0.0 até 1.0 no inferno colormap)
                normalized_intensity = min(intensity / 80.0, 1.0) # 80 costuma ser uma força alta
                bar.set_color(cmap(normalized_intensity))
                
        return bars

    # Intervalo de 50ms para 20FPS fluidos
    anim = animation.FuncAnimation(fig, animate, blit=False, interval=50, cache_frame_data=False)

    def on_close(event):
        p.terminate()
        print("Janela fechada, processos encerrados.")

    fig.canvas.mpl_connect('close_event', on_close)

    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close()
        p.terminate()
        quit()
