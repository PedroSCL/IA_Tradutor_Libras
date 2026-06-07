"""
Etapa 1 - Coleta de dados
=========================

Este script abre a webcam, captura frames com MediaPipe Holistic e salva
as sequências de landmarks em disco, organizadas por classe e por sequência.

O objetivo principal é construir o conjunto bruto de treinamento para o
tradutor de Libras sem sobrescrever coletas anteriores.

Uso:
    python src/data_collection/collect_data.py
"""

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CLASSES, DATA_RAW,
    SEQUENCES_PER_CLASS, FRAMES_PER_SEQUENCE,
)

try:
    import mediapipe as mp
except ImportError:
    print("[ERRO] mediapipe não instalado. Execute: pip install mediapipe")
    sys.exit(1)

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils


def extract_keypoints(results) -> np.ndarray:
    """Converte o resultado do MediaPipe em um vetor numérico fixo por frame.

    Cada frame pode conter três conjuntos de landmarks:
    - pose: 33 pontos do corpo
    - left_hand: 21 pontos da mão esquerda
    - right_hand: 21 pontos da mão direita

    Se algum conjunto não for detectado, o script preenche o espaço com zeros
    para manter sempre o mesmo tamanho de saída.
    """
    # Transforma os landmarks da pose em uma lista [x, y, z] para cada ponto.
    pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
                    dtype=np.float32).flatten() if results.pose_landmarks else np.zeros(33 * 3, dtype=np.float32)
    # Faz a mesma conversão para a mão esquerda.
    lh   = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                    dtype=np.float32).flatten() if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
    # Faz a mesma conversão para a mão direita.
    rh   = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                    dtype=np.float32).flatten() if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
    # Junta tudo em um único vetor de características para este frame.
    return np.concatenate([pose, lh, rh])


def draw_landmarks(image, results):
    """Desenha os landmarks detectados sobre a imagem exibida na tela.

    Isso é útil para o operador visualizar se o MediaPipe está reconhecendo
    corretamente o corpo e as mãos durante a coleta.
    """
    # Desenha a pose apenas quando ela foi detectada.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10),   thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121),  thickness=2, circle_radius=2),
        )
    # Desenha a mão esquerda apenas quando ela foi detectada.
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76),  thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    # Desenha a mão direita apenas quando ela foi detectada.
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


def get_next_index(cls_path: str) -> int:
    """Retorna o próximo índice disponível sem sobrescrever coletas anteriores.

    A estrutura esperada é:
    data/raw/<classe>/<sequencia>/keypoints.npy

    Então o script procura pastas numéricas já existentes e continua a partir
    do maior índice encontrado.
    """
    existing = [
        int(d) for d in os.listdir(cls_path)
        if os.path.isdir(os.path.join(cls_path, d)) and d.isdigit()
    ]
    return max(existing) + 1 if existing else 0


def collect():
    """Executa a rotina completa de coleta de dados pela webcam.

    O fluxo geral é:
    1. abrir a câmera;
    2. inicializar o MediaPipe Holistic;
    3. percorrer cada classe definida em config.py;
    4. gravar várias sequências por classe;
    5. salvar cada sequência como um arquivo .npy.
    """
    # Abre a câmera padrão do sistema.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Não foi possível acessar a câmera.")
        return

    print("\n[INFO] Gravando novas sequências SEM sobrescrever as existentes\n")

    # O contexto do Holistic mantém o modelo carregado durante toda a coleta.
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:

        # Percorre todas as classes de sinais definidas no arquivo de configuração.
        for cls_idx, cls in enumerate(CLASSES):
            # Pasta onde as sequências desta classe serão armazenadas.
            cls_path  = os.path.join(DATA_RAW, cls)
            os.makedirs(cls_path, exist_ok=True)

            # Descobre o próximo índice livre para continuar a coleta sem apagar dados.
            start_idx = get_next_index(cls_path)
            end_idx   = start_idx + SEQUENCES_PER_CLASS

            print(f"\n{'='*50}")
            print(f"  Sinal: {cls.upper()}  ({cls_idx + 1}/{len(CLASSES)})")
            print(f"  Sequências existentes: {start_idx} | Gravando: {start_idx} → {end_idx - 1}")
            print(f"{'='*50}")

            # Cada sequência representa uma gravação completa do mesmo sinal.
            for seq in range(start_idx, end_idx):
                seq_num = seq - start_idx + 1

                # Mostra uma contagem regressiva para o usuário se posicionar.
                for countdown in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.putText(frame, f"SINAL: {cls.upper()}  Seq {seq_num}/{SEQUENCES_PER_CLASS}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Iniciando em {countdown}...",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.imshow("Coleta de Dados — Libras", frame)
                    cv2.waitKey(1000)

                # Guarda os vetores de landmarks de todos os frames desta sequência.
                frames_data = []
                for frame_num in range(FRAMES_PER_SEQUENCE):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Converte a imagem de BGR para RGB, que é o formato esperado pelo MediaPipe.
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Desabilita escrita temporariamente para ganhar desempenho.
                    img_rgb.flags.writeable = False
                    results = holistic.process(img_rgb)
                    # Reabilita escrita para permitir novas conversões e desenho.
                    img_rgb.flags.writeable = True
                    # Volta a imagem para BGR para exibição com OpenCV.
                    frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                    # Desenha os landmarks para acompanhamento visual.
                    draw_landmarks(frame, results)
                    # Extrai e armazena o vetor numérico do frame.
                    frames_data.append(extract_keypoints(results))

                    cv2.putText(frame, f"GRAVANDO: {cls.upper()}  [{frame_num+1}/{FRAMES_PER_SEQUENCE}]",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Coleta de Dados — Libras", frame)

                    # Tecla q interrompe a coleta imediatamente.
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                # Cria a pasta da sequência e salva o array completo em .npy.
                save_dir  = os.path.join(cls_path, str(seq))
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "keypoints.npy")
                np.save(save_path, np.array(frames_data, dtype=np.float32))
                print(f"  [SALVO] {cls}/{seq} → {save_path}")

    # Libera a câmera e fecha todas as janelas ao terminar.
    cap.release()
    cv2.destroyAllWindows()
    print("\n[CONCLUÍDO] Coleta de dados finalizada!")


if __name__ == "__main__":
    # Executa a coleta somente quando o arquivo é chamado diretamente.
    collect()