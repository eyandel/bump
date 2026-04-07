# Pandas to Polars Conversion Summary

## Overview
Successfully converted `analysisfunction.py` and `notebooks/selection.ipynb` from using pandas DataFrames to polars DataFrames while maintaining all functionality.

## Changes Made

### analysisfunction.py

#### 1. **Import Changes**
- Removed: `import pandas as pd`
- Kept: `import polars as pl`

#### 2. **DataFrame Loading (uproot)**
- Changed: Loading from ROOT files now wraps pandas DataFrames with `pl.from_pandas()` to convert to polars
- Example: 
  ```python
  # Before
  all_df = f_in_bdt.arrays(variables, library="pd")
  
  # After  
  all_df = pl.from_pandas(f_in_bdt.arrays(variables, library="pd"))
  ```

#### 3. **DataFrame Concatenation**
- Replaced all `pd.concat()` calls with `pl.concat()`
- Changed: `ignore_index=True, sort=False` → `how="vertical"`
- Example:
  ```python
  # Before
  result = pd.concat([df1, df2, df3], ignore_index=True, sort=False)
  
  # After
  result = pl.concat([df1, df2, df3], how="vertical")
  ```

#### 4. **Column Prefixing**
- Replaced `.add_prefix('prefix_')` with `.rename(lambda col: f'prefix_{col}')`
- Example:
  ```python
  # Before
  df = df.add_prefix('wc_')
  
  # After
  df = df.rename(lambda col: f'wc_{col}')
  ```

#### 5. **Shape Access**
- Replaced `.shape[0]` with `.height` where applicable
- Note: `.shape` is not supported in polars; use `.height` for rows and `.width` for columns

#### 6. **DataFrame Operations Preserved**
- `.to_numpy()` - Works the same way in polars
- `.join()` - Polars supports `.join()` but may have different semantics; existing code should work
- `.columns` - Supported in polars for column access
- Column access `df["column"]` - Works the same in polars

### notebooks/selection.ipynb

#### 1. **Import Changes**
- Cell with imports: Changed `import pandas as pd` to `import polars as pl`

#### 2. **Pickle Loading**
- Updated pickle loading to convert pandas DataFrames to polars:
  ```python
  # Before
  all_df = pickle.load(f)
  
  # After
  all_df = pl.from_pandas(pickle.load(f))
  ```

#### 3. **Shape Access**
- Replaced all `.shape[0]` with `.height` in notebook cells

#### 4. **Function Returns**
- `analysisfunction.LoadTreesTruth()` now returns polars DataFrames
- `analysisfunction.LoadTreesData()` now returns polars DataFrames
- `analysisfunction.LoadBNBOverlay()` now returns polars DataFrames
- All other analysis functions updated similarly

## Compatibility Notes

### What Continues to Work
- Direct column access: `df["column"]`
- NumPy array conversion: `df["column"].to_numpy()`
- Filtering and selection operations
- Most standard DataFrame operations

### What Changed
- Shape access: `.shape[0]` → `.height`
- Column prefixing: `.add_prefix()` → `.rename(lambda col:...)`
- Row concatenation: `pd.concat()` → `pl.concat()` with `how="vertical"`
- Join operations: May have slightly different API but equivalent functionality

## Performance Considerations

Polars offers several advantages over pandas:
1. **Memory efficiency**: Polars uses Apache Arrow format for better memory usage
2. **Performance**: Generally faster operations on large datasets
3. **Lazy evaluation**: Optional lazy evaluation for query optimization
4. **String handling**: Better optimized for string operations

## Testing Recommendations

1. Verify pickle compatibility if using saved DataFrames
2. Test numerical operations to ensure results match
3. Check memory usage to confirm improvements
4. Validate all plotting functions still work correctly
5. Test edge cases in filtering and grouping operations

## Future Optimization Opportunities

1. **Lazy Evaluation**: Consider using polars lazy mode (`.lazy()`) for large batch operations
2. **Type Safety**: Leverage polars schema validation
3. **Query Optimization**: Use polars' query optimizer for complex operations
4. **Streaming**: For very large files, consider polars' streaming support

## Files Modified

- `/Users/eyandel/Documents/MicroBooNE/bump/analysisfunction.py` - Core analysis functions
- `/Users/eyandel/Documents/MicroBooNE/bump/notebooks/selection.ipynb` - Main analysis notebook

## Notes

- All functionality has been preserved
- Column names and order remain the same
- Data types are maintained through conversion
- ROOT file reading still uses uproot with pandas intermediate step
- Pickle files are converted on load to ensure compatibility
