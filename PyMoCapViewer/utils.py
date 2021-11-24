def create_xy_points(
        x_coord: int,
        y_coord: int,
        cell_size: float,
        offset: float
):
    p1 = [x_coord * cell_size - offset, y_coord * cell_size - offset, 0]
    p2 = [(x_coord * cell_size + cell_size) - offset, y_coord * cell_size - offset, 0]
    p3 = [(x_coord * cell_size + cell_size) - offset, (y_coord * cell_size + cell_size) - offset, 0]
    p4 = [x_coord * cell_size - offset, (y_coord * cell_size + cell_size) - offset, 0]
    return p1, p2, p3, p4


def create_yz_points(
        x_coord: int,
        y_coord: int,
        cell_size: float,
        offset: float
):
    p1 = [0, x_coord * cell_size - offset, y_coord * cell_size - offset]
    p2 = [0, (x_coord * cell_size + cell_size) - offset, y_coord * cell_size - offset]
    p3 = [0, (x_coord * cell_size + cell_size) - offset, (y_coord * cell_size + cell_size) - offset]
    p4 = [0, x_coord * cell_size - offset, (y_coord * cell_size + cell_size) - offset]
    return p1, p2, p3, p4


def create_xz_points(
        x_coord: int,
        y_coord: int,
        cell_size: float,
        offset: float
):
    p1 = [x_coord * cell_size - offset, 0, y_coord * cell_size - offset]
    p2 = [(x_coord * cell_size + cell_size) - offset, 0, y_coord * cell_size - offset]
    p3 = [(x_coord * cell_size + cell_size) - offset, 0, (y_coord * cell_size + cell_size) - offset]
    p4 = [x_coord * cell_size - offset, 0, (y_coord * cell_size + cell_size) - offset]
    return p1, p2, p3, p4
