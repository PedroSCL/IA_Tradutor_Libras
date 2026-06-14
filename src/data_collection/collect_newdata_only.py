"""
Coleta de Dados — Apenas Sinais Novos
======================================
Grava sequências SOMENTE para as classes que ainda não possuem
nenhum landmark em data/raw/. Classes já existentes são ignoradas.

Uso:
    python src/data_collection/collect_new_only.py
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
    pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
                    dtype=np.float32).flatten() if results.pose_landmarks else np.zeros(33 * 3, dtype=np.float32)
    lh   = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                    dtype=np.float32).flatten() if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
    rh   = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                    dtype=np.float32).flatten() if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)
    return np.concatenate([pose, lh, rh])


def draw_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10),   thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121),  thickness=2, circle_radius=2),
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76),  thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


def has_landmarks(cls_path: str) -> bool:
    """Retorna True se a pasta já tiver pelo menos uma sequência com keypoints.npy."""
    if not os.path.exists(cls_path):
        return False
    for d in os.listdir(cls_path):
        subdir = os.path.join(cls_path, d)
        if os.path.isdir(subdir) and d.isdigit():
            if os.path.exists(os.path.join(subdir, "keypoints.npy")):
                return True
    return False


def collect_new_only():
    # Filtra apenas as classes que ainda não têm landmarks
    new_classes = [
        cls for cls in CLASSES
        if not has_landmarks(os.path.join(DATA_RAW, cls))
    ]

    if not new_classes:
        print("[INFO] Todos os sinais já possuem landmarks. Nada a gravar.")
        return

    print(f"\n[INFO] Sinais que já têm landmarks (serão ignorados):")
    for cls in CLASSES:
        if cls not in new_classes:
            print(f"  ✓ {cls}")

    print(f"\n[INFO] Sinais NOVOS que serão gravados agora:")
    for cls in new_classes:
        print(f"  → {cls}")

    input("\nPressione ENTER para começar a gravação...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Não foi possível acessar a câmera.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:

        for cls_idx, cls in enumerate(new_classes):
            cls_path = os.path.join(DATA_RAW, cls)
            os.makedirs(cls_path, exist_ok=True)

            print(f"\n{'='*50}")
            print(f"  Sinal NOVO: {cls.upper()}  ({cls_idx + 1}/{len(new_classes)})")
            print(f"  Gravando sequências 0 → 69")
            print(f"{'='*50}")

            for seq in range(70):
                # Contagem regressiva
                for countdown in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.putText(frame, f"SINAL: {cls.upper()}  Seq {seq+1}/{70}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Iniciando em {countdown}...",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.imshow("Coleta — Sinais Novos", frame)
                    cv2.waitKey(1000)

                # Grava os frames
                frames_data = []
                for frame_num in range(FRAMES_PER_SEQUENCE):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_rgb.flags.writeable = False
                    results = holistic.process(img_rgb)
                    img_rgb.flags.writeable = True
                    frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                    draw_landmarks(frame, results)
                    frames_data.append(extract_keypoints(results))

                    cv2.putText(frame, f"GRAVANDO: {cls.upper()}  [{frame_num+1}/{FRAMES_PER_SEQUENCE}]",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Coleta — Sinais Novos", frame)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                # Salva a sequência
                save_dir  = os.path.join(cls_path, str(seq))
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "keypoints.npy")
                np.save(save_path, np.array(frames_data, dtype=np.float32))
                print(f"  [SALVO] {cls}/{seq} → {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[CONCLUÍDO] Coleta de sinais novos finalizada!")


if __name__ == "__main__":
    collect_new_only()