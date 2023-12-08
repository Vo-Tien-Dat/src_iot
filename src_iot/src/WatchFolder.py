import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class EventHandler(FileSystemEventHandler):
    def __init__(self, callback) -> None:
        super().__init__()
        self.callback = callback

    def on_created(self, event):
        print(f"Thêm tệp: {event.src_path}")
        self.callback(src_path=event.src_path)

class WatchFolder():
    def __init__(self, src_path, callback) -> None:
        print(f" Đang lắng nghe thư mục {src_path}")

        self.src_path = src_path
        self.callback = callback
        event_handler = EventHandler(callback=self.callback)
        observer = Observer()
        observer.schedule(event_handler, path=src_path, recursive=True)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()

        observer.join()
