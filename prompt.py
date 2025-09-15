def proper_child_prompt(parent_width, parent_height, child_width, child_height):

    coverage_ratio = (child_width * child_height) / (parent_width * parent_height) * 100
    
    return f"""
    You are a developer cleaning up UI component hierarchy before code generation.

    VISUAL ANALYSIS:
    - The ENTIRE IMAGE shows the PARENT component content
    - BLUE box = CHILD component (the only child within this parent)
    - Look at what's inside vs outside the blue box

    DECISION NEEDED:
    Is this child a PROPER SEMANTIC CHILD or UNNECESSARY WRAPPER?

    PROPER SEMANTIC CHILD:
    - Child (blue box) contains distinct content different from the surrounding parent area
    - Represents a meaningful UI section (card body, header content, form area)
    - Would be a separate <div> or component in React
    - You can see meaningful content outside the blue box within the parent

    UNNECESSARY WRAPPER:
    - Child (blue box) contains the exact same content as the entire parent image
    - No meaningful content visible outside the blue box
    - Just adds padding, margins, or styling
    - Should be skipped to find real children

    **IMPORTANT GUIDELINES:**

    1. ATOMIC COMPONENT RULE:
    - DO NOT divide very small elements that can be built as single components
    - Avoid going to extremes like individual icons, buttons, or simple text elements
    - If the component feels "atomic" enough for coding, don't divide further

    2. REPEATED COMPONENTS RULE:
    - If you see repeated/similar components within the container (like list items, cards, menu items)
    - There's usually NO NEED for further division
    - These can be handled as array/loop components in code

    VISUAL CLUES:
    - Is there meaningful content outside the blue box?
    - Does the blue box represent a distinct section within the parent?
    - Is this component small/simple enough to code as one piece?
    - Are there repeated patterns that suggest this is a list/collection?

    COMPONENT INFO:
    Parent: {parent_width}x{parent_height}
    Child: {child_width}x{child_height}  
    Coverage: {coverage_ratio}%

    RESPONSE (JSON): Always provide a JSON response and delimit with '```json' and '```'.
    ```json
    {{
        "is_proper_child": true or false,
        "explanation": "What you see visually that justifies this decision"
    }}
    ```

    Think: Would I create a separate component for this child in my code?
    """



def heirarchy_end_prompt(comp_width, comp_height):
    return f"""
    You are a UI developer deciding if a component needs further subdivision for code generation.

    VISUAL INPUTS:
    - Image 1: Full UX screen with RED highlighted Bounding Box around the component, with Bounding Box Label as 'BBox' (shows context) 
    - Image 2: Cropped view of just the component content (shows internal detail)

    DECISION NEEDED:
    Should this component be DIVIDED FURTHER or is it ready for code generation?
    **IMPORTANT CONSIDERATIONS:**
    1. If the component is so small that only the bounding box label ('BBox') is visible in Image 1, but the bounding box itself is not clearly visible, then the component should **not** be further divided. In such cases, treat it as "READY FOR CODE."
    2. If the bounding box height is so small that the red bounding box appears as a very thin line in Image 1, then the component should **not** be further divided. In such cases, treat it as "READY FOR CODE."
    
    ASSESSMENT:
    Look at the component's internal structure and ask:
    - Do you see multiple distinct elements that should be separate components?
    - Are there clear functional boundaries within this area?
    - Would a developer naturally split this into smaller parts?
    - Or does it feel like one cohesive, atomic component?

    DIVIDE_FURTHER means:
    - Multiple distinct UI sections visible
    - Clear separation of concerns within the component  
    - Natural breaking points that would improve code organization
    - Parts that could be reused independently

    READY FOR CODE means:
    - Feels like one cohesive unit
    - Elements work together as a single component
    - No obvious need to break it down further
    - Atomic enough for clean code generation

    COMPONENT Dimensions: {comp_width}x{comp_height}

    RESPONSE (JSON): Always provide a JSON response and delimit with '```json' and '```'.
    ```json
    {{
        "should_divide_further": true or false,
        "explanation": "What you see in the component that led to this decision"
    }}
    ```

    Think: Would I code this as one component or break it into smaller parts?
    """