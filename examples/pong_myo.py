import asyncio
import multiprocessing
import logging
import pygame
import sys
import random

from myo import MyoClient
from myo.types import ClassifierMode, EMGMode, IMUMode

# ==========================================
# 1. SETUP DO DL-MYO (USANDO IMU / GIROSCÓPIO)
# ==========================================
class PongClient(MyoClient):
    async def on_imu_data(self, imu):
        # O Giroscópio possui 3 eixos (x, y, z).
        # O eixo [0] geralmente é a rotação do antebraço (pronação/supinação).
        # Enviamos a velocidade de rotação atual para a fila.
        gyro_x = imu.gyroscope[0]
        self.q.put(gyro_x)

    # --- Métodos abstratos obrigatórios ignorados ---
    async def on_classifier_event(self, ce): pass
    async def on_emg_data(self, emg): pass
    async def on_aggregated_data(self, ad): pass
    async def on_motion_event(self, me): pass
    async def on_emg_data_aggregated(self, eds): pass
    async def on_fv_data(self, fvd): pass

async def run_myo_bluetooth(q, mac_address):
    logging.info("Conectando à Myo via Bluetooth...")
    client = await PongClient.with_device(mac=mac_address)
    client.q = q
    
    # ATENÇÃO AQUI: Desligamos o classificador e ligamos o SEND_DATA do IMU
    await client.setup(
        classifier_mode=ClassifierMode.DISABLED,
        emg_mode=EMGMode.NONE, 
        imu_mode=IMUMode.SEND_DATA
    )
    
    await client.start()
    logging.info("Myo IMU Pronta! Gire o braço (como se estivesse acelerando uma moto) para mover a raquete sem delay.")
    
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
# 2. SETUP DO JOGO PING PONG (PYGAME)
# ==========================================
if __name__ == '__main__':
    MAC = "E8:59:E3:95:37:F1"

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(q, MAC))
    p.start()

    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Myo Pong - Zero Delay (IMU)")
    clock = pygame.time.Clock()

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    paddle_width, paddle_height = 15, 100
    ball_size = 15

    player_x, player_y = 30, HEIGHT//2 - paddle_height//2
    ai_x, ai_y = WIDTH - 30 - paddle_width, HEIGHT//2 - paddle_height//2
    ball_x, ball_y = WIDTH//2, HEIGHT//2

    player_speed = 0
    ai_speed = 6
    ball_speed_x, ball_speed_y = 5 * random.choice((1, -1)), 5 * random.choice((1, -1))

    score_player = 0
    score_ai = 0
    font = pygame.font.Font(None, 74)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    p.terminate()
                    sys.exit()

            # Checa se o processo Bluetooth morreu inesperadamente
            if not p.is_alive():
                print("\n[ERRO] O processo Bluetooth foi encerrado. O jogo será fechado.")
                pygame.quit()
                sys.exit()

            # LER COMANDOS IMU
            # Pegamos sempre o último dado da fila para garantir resposta instantânea
            latest_gyro = None
            while not q.empty():
                latest_gyro = q.get()
            
            if latest_gyro is not None:
                # O multiplicador (20) define a SENSibilidade.
                # Se estiver muito rápido, diminua para 10. Se estiver lento, aumente para 30.
                # Como o giroscópio lê velocidade angular, se você parar de girar o braço (velocidade 0), a raquete para.
                player_speed = latest_gyro * 0.8

            # Desaceleração natural (fricção) para a raquete não deslizar para sempre
            player_speed *= 0.2 

            player_y += player_speed
            if player_y <= 0: player_y = 0
            if player_y >= HEIGHT - paddle_height: player_y = HEIGHT - paddle_height

            # IA Simples
            if ai_y + paddle_height//2 < ball_y:
                ai_y += ai_speed
            elif ai_y + paddle_height//2 > ball_y:
                ai_y -= ai_speed
            if ai_y <= 0: ai_y = 0
            if ai_y >= HEIGHT - paddle_height: ai_y = HEIGHT - paddle_height

            ball_x += ball_speed_x
            ball_y += ball_speed_y

            if ball_y <= 0 or ball_y >= HEIGHT - ball_size:
                ball_speed_y *= -1

            player_rect = pygame.Rect(player_x, player_y, paddle_width, paddle_height)
            ai_rect = pygame.Rect(ai_x, ai_y, paddle_width, paddle_height)
            ball_rect = pygame.Rect(ball_x, ball_y, ball_size, ball_size)

            if ball_rect.colliderect(player_rect) and ball_speed_x < 0:
                ball_speed_x *= -1.1
            if ball_rect.colliderect(ai_rect) and ball_speed_x > 0:
                ball_speed_x *= -1.1

            if ball_x <= 0:
                score_ai += 1
                ball_x, ball_y = WIDTH//2, HEIGHT//2
                ball_speed_x = 5
            elif ball_x >= WIDTH:
                score_player += 1
                ball_x, ball_y = WIDTH//2, HEIGHT//2
                ball_speed_x = -5

            screen.fill(BLACK)
            pygame.draw.rect(screen, WHITE, player_rect)
            pygame.draw.rect(screen, WHITE, ai_rect)
            pygame.draw.ellipse(screen, WHITE, ball_rect)
            pygame.draw.aaline(screen, WHITE, (WIDTH//2, 0), (WIDTH//2, HEIGHT))

            player_text = font.render(str(score_player), True, WHITE)
            ai_text = font.render(str(score_ai), True, WHITE)
            screen.blit(player_text, (WIDTH//4, 20))
            screen.blit(ai_text, (WIDTH*3//4, 20))

            pygame.display.flip()
            clock.tick(60)

    except KeyboardInterrupt:
        pygame.quit()
        p.terminate()
        sys.exit()