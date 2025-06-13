import cv2
import time
import os


def main():
    rtsp_url = os.environ.get('DOORBELL_RTSP_URL', 'rtsp://user:pass@host:554/Streaming/Channels/101/')
    pipeline = (
        f"rtspsrc location={rtsp_url} latency=50 ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Failed to open GStreamer pipeline.")
        return

    # Track time between frames to compute FPS
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received.")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - last_time)
        last_time = current_time

        # Draw FPS in the center of the frame
        fps_text = f"FPS: {fps:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_size, _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
        text_width, text_height = text_size

        height, width = frame.shape[:2]
        # Compute the coordinates for centered text
        text_x = (width - text_width) // 2
        text_y = (height + text_height) // 2

        cv2.putText(
            frame,
            fps_text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 0),  # Green text color
            thickness
        )

        cv2.imshow("RTSP Stream (GStreamer)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
