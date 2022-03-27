# 3D Slicer plugin

Additional register of the table
## Preparation of the environment
See [Model Offset](../ModelOffset/Readme.md).

The only change - one can use `Developer Tools for Extensions` to load existing (i.e. this one) plugin to 3D Slicer.

## Limitations 
1. Control points on the table are in fixed order:
   1. lower right
   2. lowe left
   3. upper left
2. CT orientation is not taken into account; therefore it should have a singular matrix of rotation.
3. If the control points are already attached to a transform, their physical coordinates (global positions) are used to calculations, not their local values. Probably there should be a check-box to allow choosing either of these.
4. For translation, the number and order of both point sets must be correct. 
5. Both optimizations (rotational and translational) are very poor. 



## experiments
- Punkty nawigacja - w układzie współrzędnych trackera
- Punkty stołu - w układzie współrzędnych trackera
- polaris to dicom - to tak naprawdę reference to dicom (powstało z kalibracji punktów)