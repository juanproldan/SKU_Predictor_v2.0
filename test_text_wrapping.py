#!/usr/bin/env python3
"""
Test the text wrapping logic we implemented for the UI fix.
"""

def test_text_wrapping():
    """Test our text wrapping implementation"""
    
    def wrap_text(display_desc, max_chars_per_line=25):
        """Same logic as implemented in main_app.py"""
        if len(display_desc) > max_chars_per_line:
            # Simple word wrapping - split at spaces when possible
            words = display_desc.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line + " " + word) <= max_chars_per_line:
                    current_line = current_line + " " + word if current_line else word
                else:
                    if current_line:
                        lines.append(current_line)
                        current_line = word
                    else:
                        # Word is longer than max_chars_per_line, truncate it
                        lines.append(word[:max_chars_per_line-3] + "...")
                        current_line = ""
            
            if current_line:
                lines.append(current_line)
            
            # Limit to 2 lines maximum
            if len(lines) > 2:
                lines = lines[:2]
                lines[1] = lines[1][:max_chars_per_line-3] + "..."
            
            wrapped_text = "\n".join(lines)
        else:
            wrapped_text = display_desc
        
        return wrapped_text
    
    # Test cases
    test_cases = [
        "LUNA ESPEJO DERECHO",  # 19 chars - should not wrap
        "DIRECCIONAL ESPEJO DERECHO",  # 26 chars - should wrap
        "PUERTA TRASERA DELANTERA DERECHA",  # 32 chars - should wrap
        "CLIPS FAROLA IZQUIERDA - GUIA FAROLA IZQUIERDA",  # 46 chars - should wrap to 2 lines
        "SOPORTE CENTRAL PARAGOLPES DELANTERO",  # 36 chars - should wrap
        "VERYLONGWORDTHATCANNOTBEWRAPPEDPROPERLY",  # Single long word - should truncate
    ]
    
    print("🧪 Testing text wrapping logic...")
    print("📏 Max characters per line: 25")
    print("📄 Max lines: 2")
    print()
    
    for i, test_desc in enumerate(test_cases, 1):
        wrapped = wrap_text(test_desc)
        lines = wrapped.split('\n')
        
        print(f"Test {i}: {test_desc} ({len(test_desc)} chars)")
        print(f"Result: {repr(wrapped)}")
        print(f"Lines: {len(lines)}")
        for j, line in enumerate(lines, 1):
            print(f"  Line {j}: '{line}' ({len(line)} chars)")
        print()

if __name__ == "__main__":
    test_text_wrapping()
