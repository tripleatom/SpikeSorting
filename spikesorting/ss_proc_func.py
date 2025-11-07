from pathlib import Path


def get_sortout_folder(default_sortout=r"\\10.129.151.108\xieluanlabs\xl_cl\code\sortout"):
    """Get and validate the sortout folder path."""
    
    print(f"\nDefault output folder: {default_sortout}")
    
    while True:
        response = input("Press Enter to accept or type a new path: ").strip()
        
        if not response:  # User pressed Enter
            sortout = default_sortout
        else:
            # Clean up the input
            sortout = response.strip('"').strip("'")
        
        # Validate the path
        sortout_path = Path(sortout)
        
        try:
            if sortout_path.exists():
                if sortout_path.is_dir():
                    print(f"✓ Using existing folder: {sortout_path}")
                    return sortout_path  # Return Path object
                else:
                    print(f"Error: {sortout_path} exists but is not a folder.")
                    continue
            else:
                create = input(
                    f"Folder doesn't exist. Create '{sortout_path}'? (y/n): ").strip().lower()
                if create == 'y':
                    sortout_path.mkdir(parents=True, exist_ok=True)
                    print(f"✓ Created folder: {sortout_path}")
                    return sortout_path  # Return Path object
                else:
                    print("Please enter a different path.")
        except Exception as e:
            print(f"Error accessing path: {e}")
            print("Please enter a valid path.")