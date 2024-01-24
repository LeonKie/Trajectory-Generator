# Trajectory Generation from Drawing

This Python script utilizes Polynomial Regression to create a trajectory from a drawing on a canvas. The script captures mouse events to draw either rectangles or curves on the canvas. The drawn points are then used to generate polynomial coefficients for both X and Y coordinates. The generated trajectory can be visualized as a curve.

## Usage

Run the script and draw on the canvas:

```bash
python script_name.py
```

Press 'm' to toggle between drawing rectangles and curves. Press 'Esc' to exit.

## Dependencies
- OpenCV
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

## Functions
- `createImg():`  Initializes the canvas and captures mouse events for drawing.
- `prepCoor(pos):` Prepares coordinates for polynomial regression.
- `getcoef(lis, deg=5):` Performs polynomial regression and returns coefficients for X and Y.
- `Polynomial:` Class representing a polynomial function.
- `curve:` Class for representing and plotting curves.
- `set_Curve(coeff_X, coeff_Y):` Sets up a curve with given X and Y coefficients.
- `plotcurve(curve):` Plots and displays the generated curve.
- `drawTrack(showPlot=True, curve=None):` Main function for drawing and generating trajectories.


## Example
```python
curve = drawTrack()
```
This will display the drawn trajectory and its polynomial representation.
