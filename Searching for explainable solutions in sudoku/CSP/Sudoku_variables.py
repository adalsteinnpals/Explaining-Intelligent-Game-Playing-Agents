from .variable import Variable


class Sudoku_Vars:
    def __init__(self, board: list):
        self.errors = False
        if len(board) != 9:
            self.errors = True
        if len(board[0]) != 9:
            self.errors = True

        self.board = [[None]*9 for _ in range(9)]
        for y in range(9):
            for x in range(9):
                if board[y][x] != 0:
                    self.board[y][x] = Variable(
                        f"{chr(y+65)}{x+1}", [board[y][x]], True, board[y][x])
                else:
                    self.board[y][x] = Variable(
                        f"{chr(y+65)}{x+1}", [1, 2, 3, 4, 5, 6, 7, 8, 9], False)

    def print_board(self):
        print('-'*19)
        for y in range(9):
            print("|", end='')
            for x in range(9):
                print("{}|".format(
                    self.board[y][x].value if self.board[y][x].status else "."), end='')
            print()
            print('-'*19)

    def get_row(self, row_n: int) -> list:
        return self.board[row_n]

    def get_col(self, col_n: int) -> list:
        return [col[col_n] for col in self.board]

    def get_box(self, x: int, y: int) -> list:
        out_box = []
        for dy in range(0, 3):
            for dx in range(0, 3):
                out_box.append(self.board[dy+(3*y)][dx+(3*x)])
        return out_box

    def get_unit(self, x: int, y: int) -> int:
        return self.board[y][x]
