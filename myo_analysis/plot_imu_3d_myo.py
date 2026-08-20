import asyncio
import multiprocessing
import queue
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# Import necessário para o 3D funcionar no matplotlib (mesmo que não seja usado diretamente)
from mpl_toolkits.mplot3d import Axes3D 

from myo import MyoClient
from myo.types import EMGMode, IMUMode, ClassifierMode

# ==========================================
# 1. SETUP DO DL-MYO (BLUETOOTH ASSÍNCRONO)
# ==========================================
class IMUClient(MyoClient):
    async def on_imu_data(self, imu):
        # Envia apenas o quaternion para a fila: (w, x, y, z)
        # O quaternion descreve a orientação da pulseira no espaço 3D
        self.q.put((imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z))

    # Ignora outros tipos de dados
    async def on_aggregated_data(self, ad): pass
    async def on_classifier_event(self, ce): pass
    async def on_motion_event(self, me): pass
    async def on_emg_data_aggregated(self, eds): pass
    async def on_fv_data(self, fvd): pass
    async def on_emg_data(self, emg): pass

async def run_myo_bluetooth(q, mac_address):
    logging.info("Conectando à Myo via Bluetooth...")
    client = await IMUClient.with_device(mac=mac_address)
    client.q = q
    
    await client.setup(
        classifier_mode=ClassifierMode.DISABLED,
        emg_mode=EMGMode.NONE, 
        imu_mode=IMUMode.SEND_DATA # Apenas IMU ativado!
    )
    
    await client.start()
    logging.info("Começando a receber os quaternions do IMU!")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await client.stop()
        await client.disconnect()

def worker(q, mac_address):
    try:
        asyncio.run(run_myo_bluetooth(q, mac_address))
    except Exception as e:
        print(f"\n[ERRO BLUETOOTH] Falha na conexão: {e}")
        print("Dica: Verifique se a pulseira está ligada, recarregada e próxima ao computador.")


# ==========================================
# 2. SETUP DO PLOT 3D (MATPLOTLIB)
# ==========================================
def quat_to_matrix(w, x, y, z):
    """ Converte um quaternion para uma matriz de rotação 3x3 """
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

if __name__ == '__main__':
    MAC = "E8:59:E3:95:37:F1"  # Seu MAC

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(q, MAC))
    p.start()

    print("Aguardando conexão com a pulseira...")
    while q.empty():
        if not p.is_alive():
            print("\nEncerrando: O processo Bluetooth falhou antes de enviar dados.")
            quit()

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 8))
    fig.canvas.manager.set_window_title("Myo IMU 3D Orientation")
    ax = fig.add_subplot(111, projection='3d')
    
    # Eixos originais que representarão a Myo
    basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Inicializa as linhas dos 3 eixos X, Y e Z
    # X normalmente é a direção ao longo do braço (onde está a logo da Myo)
    line_x, = ax.plot([0, 1], [0, 0], [0, 0], color='r', linewidth=5, label='X (Direção do braço)')
    line_y, = ax.plot([0, 0], [0, 1], [0, 0], color='g', linewidth=5, label='Y')
    line_z, = ax.plot([0, 0], [0, 0], [0, 1], color='b', linewidth=5, label='Z')
    
    # Define os limites do gráfico 3D para não ficar "pulando"
    limit = 1.5
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Orientação 3D da Pulseira (IMU)")
    ax.legend(loc="upper left")

    def animate(frame):
        latest_data = None
        # Pega a orientação mais recente da fila
        while not q.empty():
            latest_data = q.get()

        if latest_data:
            w, x, y, z = latest_data
            
            # Obtém a matriz de rotação a partir do quaternion
            R = quat_to_matrix(w, x, y, z)
            
            # Multiplica a base pela rotação para girar os eixos no 3D
            rot_basis = np.dot(R, basis.T).T
            
            # Atualiza a linha do Eixo X (Vermelho)
            line_x.set_data_3d([0, rot_basis[0, 0]], [0, rot_basis[0, 1]], [0, rot_basis[0, 2]])
            # Atualiza a linha do Eixo Y (Verde)
            line_y.set_data_3d([0, rot_basis[1, 0]], [0, rot_basis[1, 1]], [0, rot_basis[1, 2]])
            # Atualiza a linha do Eixo Z (Azul)
            line_z.set_data_3d([0, rot_basis[2, 0]], [0, rot_basis[2, 1]], [0, rot_basis[2, 2]])
            
        return line_x, line_y, line_z

    anim = animation.FuncAnimation(fig, animate, blit=False, interval=40, cache_frame_data=False)

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
