import torch
import torchvision
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
from PIL import Image

from models import linear_model



class InteractivePaintingCanvas:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.setup_window()

    def setup_window(self):
        self.root = tk.Tk()
        self.root.title("AI Drawing Tester")

        self.pixel_size = 10  # Rozmiar bloku "pikselowego"
        self.grid_size = 28  # Wymiary logiczne (28x28)
        self.canvas_size = self.grid_size * self.pixel_size  # Rozmiar płótna w pikselach
        # Logiczna macierz piksela (28x28)
        self.pixel_data = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Płótno
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.pack()

        # Przycisk do czyszczenia
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.RIGHT)



        # Połączenie zdarzenia rysowania z płótnem
        self.canvas.bind("<B1-Motion>", self.paint)
        #self.canvas.bind("<Button-1>", self.reset_drawing)

    def paint(self, event):
        # Współrzędne kursora
        self.x, self.y = event.x, event.y

        # Wyznacz, który logiczny piksel został narysowany
        col = self.x // self.pixel_size
        row = self.y // self.pixel_size

        if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
            # Oblicz intensywność na podstawie odległości od środka piksela
            intensity = 1.0 - ((self.x % self.pixel_size) / self.pixel_size) * 0.1

            self.pixel_data[row, col] = intensity
            # Rysowanie na powiększonym płótnie
            x0 = col * self.pixel_size
            y0 = row * self.pixel_size
            x1 = x0 + self.pixel_size
            y1 = y0 + self.pixel_size
            color = f'#{int(255 * intensity):02x}{int(255 * intensity):02x}{int(255 * intensity):02x}'
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)
            #print("self.pixel_data",self.pixel_data)

    def reset_drawing(self, event=None):
        """Resetuje rysowanie do wartości początkowych."""
        self.pixel_data.fill(0)

    def clear_canvas(self):
        # Czyści płótno i logiczną macierz
        self.canvas.delete("all")
        self.pixel_data.fill(0)

    def get_image_data(self):
        # Zwraca logiczną macierz jako obraz PIL (28x28)
        # Skalowanie do 0-255 z uwzględnieniem intensywności
        image = Image.fromarray((self.pixel_data * 255).astype(np.float32)).convert('L')
        return image

    def predict_digit(self):
        # Pobranie danych obrazu jako tensor
        image = self.get_image_data()
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Konwersja na tensor i spłaszczenie
        image_tensor = torch.from_numpy(image_array).view(784)  # Spłaszczenie 28x28 -> 784

        # Przewidywanie (przykład)
        with torch.no_grad():
            X = image_tensor.unsqueeze(1).cuda()  # Dodanie wymiaru batcha
            z1, h1, z2, h2, z3, y_pred, mask_forw_drop = self.model.forward(X, 0)
            predict = self.model.get_predictions(y_pred)


        # Wyświetlanie obrazu do weryfikacji
        plt.imshow(image_array, cmap='gray')
        plt.title(f"Przewidziana cyfra: {predict.item()}")
        plt.show()

    def run(self):
        self.root.mainloop()




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    input_dim = 784
    output_dim = 10

    epoch = 100
    batch_size = 256
    hidden_dim = 103
    weight_scale = (2 / input_dim) ** 0.5  # inicjalizacji He
    decay = 1e-3

    # momentum
    beta = 0.95
    lr = 0.5
    l2_reg = 0
    dropout_p = 0
    mode = 'test'

    model_linear = linear_model.Three_Leyer_Model(input_dim, hidden_dim, hidden_dim, output_dim, weight_scale, l2_reg, mode, device=device)


    model_linear.load()

    paint = InteractivePaintingCanvas(model=model_linear,device=device)
    paint.run()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()