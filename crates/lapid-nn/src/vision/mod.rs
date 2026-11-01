//! Shared pieces of ViT / SigLIP-style vision towers.
//!
//! * [`image`]: decoding (PNG / JPEG), torchvision-exact antialiased resize, normalized planes.
//! * [`patches`]: resize rules (`smart_resize`, aspect-preserving patch budget) and patch
//!   cutting (merge-block or raster order).
//! * [`rope2d`]: 2-D RoPE over patch grids (row / column split, axial halves) and RoPE from
//!   host angle tables.
//! * [`pool`]: bilinear position-table interpolation and k x k average pooling.
//! * [`mask`]: decoder attention mask with bidirectional image blocks.

pub mod image;
pub mod mask;
pub mod patches;
pub mod pool;
pub mod rope2d;

pub use image::{Filter, Image};
pub use mask::image_block_mask;
pub use patches::{Patches, aspect_preserving_size, block_order_positions, patchify, patchify_raster, smart_resize};
pub use pool::{bilinear_taps, block_mean_matrix, interpolate_pos_table};
pub use rope2d::{apply_rope_tables, axial_rope, grid_rope_tables, rope_tables};
