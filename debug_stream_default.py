import cv2
import os

def main():
    # RTSP URL now loaded from .env
    rtsp_url = os.environ.get('DOORBELL_RTSP_URL', 'rtsp://user:pass@host:554/Streaming/Channels/101/')

    # Open the RTSP feed without GStreamer
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Failed to open RTSP stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received. Exiting...")
            break

        cv2.imshow("RTSP Stream (No GStreamer)", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
