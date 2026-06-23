from pydantic import BaseModel, Field

class DocumentRotation(BaseModel):
    thinking: str = Field(
        description="Internal reasoning about the orientation of the text."
    )
    rotation_angle: int = Field(
        description="The rotation angle of the text. MUST be exactly 0, 90, 180, or 270."
    )


class DocumentComposition(BaseModel):
    thinking: str = Field(
        description=(
            "Analyze the physical layout of the image. Look for visual evidence of two distinct physical pages "
            "scanned side-by-side. Key visual evidence includes: a dark central binding gutter or shadow, "
            "distinct physical page edges visible in the middle, or page numbers located on opposite outer corners. "
            "Explicitly distinguish between a 'two-page book spread' and a single page that just uses 'two-column text formatting'."
        )
    )
    is_composite_spread: bool = Field(
        description=(
            "True ONLY if the image contains multiple distinct physical pages scanned next to each other "
            "(e.g., a two-page spread). False if it is a single physical page, even if it contains multiple columns, "
            "tables, or dense text layouts."
        )
    )