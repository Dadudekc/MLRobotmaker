import tkinter as tk

class ConnectFour:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect Four")
        self.root.resizable(True, True)
        self.initialize_game()
        self.root.bind('<Configure>', self.resize)

    def initialize_game(self):
        self.grid = [[' ' for _ in range(7)] for _ in range(6)]
        self.canvases = []
        for row in range(6):
            row_list = []
            for col in range(7):
                canvas = tk.Canvas(self.root, bg='white')
                canvas.grid(row=row, column=col, sticky="nsew")
                self.root.grid_rowconfigure(row, weight=1)
                self.root.grid_columnconfigure(col, weight=1)
                canvas.bind("<Button-1>", lambda event, c=col: self.drop_disc(c))
                row_list.append(canvas)
            self.canvases.append(row_list)
        self.current_player = 'R'
        self.label = tk.Label(self.root, text="Player Red's Turn")
        self.label.grid(row=7, columnspan=7, sticky="ew")
        self.root.grid_rowconfigure(7, weight=0)

    def draw_circle(self, canvas, color):
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        size = min(canvas_width, canvas_height)
        margin = size * 0.1
        canvas.create_oval(margin, margin, size - margin, size - margin, fill=color, outline=color)

    def drop_disc(self, col):
        row = self.find_row(col)
        if row is not None:
            self.grid[row][col] = self.current_player
            color = 'red' if self.current_player == 'R' else 'yellow'
            self.draw_circle(self.canvases[row][col], color)
            if self.check_win(row, col):
                self.label.config(text=f"Player {self.current_player} Wins!")
                for row_canvases in self.canvases:
                    for canvas in row_canvases:
                        canvas.unbind("<Button-1>")
                return
            self.switch_player()

    def resize(self, event):
        for row in self.canvases:
            for canvas in row:
                self.redraw_canvas(canvas)

    def redraw_canvas(self, canvas):
        canvas.delete("all")
        col = canvas.grid_info()['column']
        row = canvas.grid_info()['row']
        if self.grid[row][col] != ' ':
            color = 'red' if self.grid[row][col] == 'R' else 'yellow'
            self.draw_circle(canvas, color)

    def find_row(self, col):
        for row in range(5, -1, -1):
            if self.grid[row][col] == ' ':
                return row
        return None

    def check_win(self, row, col):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, two diagonals
        for dr, dc in directions:
            count = 0
            for i in range(-3, 4):
                r = row + i * dr
                c = col + i * dc
                if 0 <= r < 6 and 0 <= c < 7 and self.grid[r][c] == self.current_player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0
        return False

    def switch_player(self):
        self.current_player = 'Y' if self.current_player == 'R' else 'R'
        self.label.config(text=f"Player {self.current_player}'s Turn")

def main():
    root = tk.Tk()
    game = ConnectFour(root)
    root.mainloop()

if __name__ == "__main__":
    main()
