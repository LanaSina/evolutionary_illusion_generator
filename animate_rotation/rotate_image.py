import tkinter as tk
from PIL import ImageTk
from PIL import Image

class SimpleApp(object):
    def __init__(self, master, filename, **kwargs):
        self.master = master
        self.filename = filename
        self.canvas = tk.Canvas(master, width=1000, height=1000)
        self.canvas.pack()

        self.update = self.draw().__next__
        master.after(50, self.update)

    def draw(self):
        image = Image.open(self.filename)
        angle = 0
        speed = 10 # 10
        while True:
            tkimage = ImageTk.PhotoImage(image.rotate(angle, fillcolor = "white"))
            canvas_obj = self.canvas.create_image(
                500, 500, image=tkimage)
            self.master.after_idle(self.update)
            yield
            self.canvas.delete(canvas_obj)
            angle += speed
            angle %= 360

root = tk.Tk()
app = SimpleApp(root, '../evolved_illusions/benham/00.png')
root.mainloop()