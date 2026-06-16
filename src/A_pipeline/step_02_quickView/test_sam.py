import argparse

import cv2

from src.C_quick_view.sam_segmenter import SAMSegmenter


def draw_detections(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox_xyxy"])
        label = (
            f"{detection['class_name']} "
            f"{detection['confidence']:.2f}"
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--mode", choices=["LoHa", "DoRa"], default="LoHa")
    parser.add_argument("--api-path", default="Analizador de video/API_sam3.json")
    arguments = parser.parse_args()

    capture = cv2.VideoCapture(arguments.video_path)
    success, frame = capture.read()
    capture.release()

    if not success:
        raise RuntimeError("No se pudo leer el primer frame.")

    sam = SAMSegmenter(
        mode=arguments.mode,
        api_path=arguments.api_path,
    )

    detections = []

    for prompt in ["playing field", "orange ball", "robots"]:
        results = sam.detect(frame, prompt)
        detections.extend(results)
        print(f"{prompt}: {len(results)} detecciones")

    draw_detections(frame, detections)
    cv2.imwrite("outputs/sam_first_frame.jpg", frame)

    print("Resultado guardado en outputs/sam_first_frame.jpg")


if __name__ == "__main__":
    main()