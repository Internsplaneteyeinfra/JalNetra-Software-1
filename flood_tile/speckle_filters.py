from __future__ import annotations

import ee


def to_natural(img: ee.Image) -> ee.Image:
    return ee.Image(10.0).pow(img.select(0).divide(10.0))


def to_db(img: ee.Image) -> ee.Image:
    return ee.Image(img).log10().multiply(10.0)


def refined_lee(img: ee.Image) -> ee.Image:
    """Refined Lee speckle filter (SNAP S1TBX adaptation by Guido Lemoine)."""
    weights3 = ee.List.repeat(ee.List.repeat(1, 3), 3)
    kernel3 = ee.Kernel.fixed(3, 3, weights3, 1, 1, False)

    mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3)
    variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3)

    sample_weights = ee.List(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    sample_kernel = ee.Kernel.fixed(7, 7, sample_weights, 3, 3, False)

    sample_mean = mean3.neighborhoodToBands(sample_kernel)
    sample_var = variance3.neighborhoodToBands(sample_kernel)

    gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
    gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
    gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
    gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())

    max_gradient = gradients.reduce(ee.Reducer.max())
    gradmask = gradients.eq(max_gradient)
    gradmask = gradmask.addBands(gradmask)

    directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(
        sample_mean.select(4).subtract(sample_mean.select(7))
    ).multiply(1)
    directions = directions.addBands(
        sample_mean.select(6).subtract(sample_mean.select(4)).gt(
            sample_mean.select(4).subtract(sample_mean.select(2))
        ).multiply(2)
    )
    directions = directions.addBands(
        sample_mean.select(3).subtract(sample_mean.select(4)).gt(
            sample_mean.select(4).subtract(sample_mean.select(5))
        ).multiply(3)
    )
    directions = directions.addBands(
        sample_mean.select(0).subtract(sample_mean.select(4)).gt(
            sample_mean.select(4).subtract(sample_mean.select(8))
        ).multiply(4)
    )
    directions = directions.addBands(directions.select(0).Not().multiply(5))
    directions = directions.addBands(directions.select(1).Not().multiply(6))
    directions = directions.addBands(directions.select(2).Not().multiply(7))
    directions = directions.addBands(directions.select(3).Not().multiply(8))
    directions = directions.updateMask(gradmask)
    directions = directions.reduce(ee.Reducer.sum())

    sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))
    sigma_v = (
        sample_stats.toArray()
        .arraySort()
        .arraySlice(0, 0, 5)
        .arrayReduce(ee.Reducer.mean(), [0])
    )

    rect_weights = ee.List.repeat(ee.List.repeat(0, 7), 3).cat(
        ee.List.repeat(ee.List.repeat(1, 7), 4)
    )
    diag_weights = ee.List(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ]
    )
    rect_kernel = ee.Kernel.fixed(7, 7, rect_weights, 3, 3, False)
    diag_kernel = ee.Kernel.fixed(7, 7, diag_weights, 3, 3, False)

    dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1))
    dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1))
    dir_mean = dir_mean.addBands(
        img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2))
    )
    dir_var = dir_var.addBands(
        img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2))
    )

    for i in range(1, 4):
        dir_mean = dir_mean.addBands(
            img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(
                directions.eq(2 * i + 1)
            )
        )
        dir_var = dir_var.addBands(
            img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(
                directions.eq(2 * i + 1)
            )
        )
        dir_mean = dir_mean.addBands(
            img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(
                directions.eq(2 * i + 2)
            )
        )
        dir_var = dir_var.addBands(
            img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(
                directions.eq(2 * i + 2)
            )
        )

    dir_mean = dir_mean.reduce(ee.Reducer.sum())
    dir_var = dir_var.reduce(ee.Reducer.sum())

    var_x = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigma_v)).divide(sigma_v.add(1.0))
    b = var_x.divide(dir_var)
    result = dir_mean.add(b.multiply(img.subtract(dir_mean)))
    return result.arrayFlatten([["sum"]])
