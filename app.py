"""
Fabric Cutting Optimizer - Streamlit Web App
Optimal fabric layout using Google OR-Tools
"""

import streamlit as st
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Fabric Cutting Optimizer",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #667eea;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #5568d3 0%, #6a3f8f 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pieces' not in st.session_state:
    st.session_state.pieces = [
        {'name': 'Piece_A', 'length': 50.0, 'width': 40.0, 'quantity': 2},
        {'name': 'Piece_B', 'length': 60.0, 'width': 35.0, 'quantity': 3},
    ]

if 'solution' not in st.session_state:
    st.session_state.solution = None


def optimize_fabric(pieces, fabric_width, time_limit):
    """Optimize fabric layout using OR-Tools CP-SAT solver"""
    
    # Expand pieces based on quantity
    all_pieces = []
    for p in pieces:
        for i in range(int(p['quantity'])):
            all_pieces.append({
                'length': float(p['length']),
                'width': float(p['width']),
                'name': f"{p['name']}_{i+1}" if p['quantity'] > 1 else p['name']
            })
    
    if not all_pieces:
        return None
    
    # Check if any piece exceeds fabric width
    for piece in all_pieces:
        if piece['width'] > fabric_width:
            st.error(f"❌ Error: {piece['name']} width ({piece['width']}cm) exceeds fabric width ({fabric_width}cm)!")
            return None
    
    # Estimate max length
    total_area = sum(p['length'] * p['width'] for p in all_pieces)
    max_length = int(total_area / fabric_width * 2.0)
    
    # Create model
    model = cp_model.CpModel()
    SCALE = 100
    
    fabric_width_scaled = int(fabric_width * SCALE)
    max_length_scaled = int(max_length * SCALE)
    
    # Variables
    x_vars = []
    y_vars = []
    
    for i, piece in enumerate(all_pieces):
        x = model.NewIntVar(0, fabric_width_scaled, f"x_{i}")
        x_vars.append(x)
        y = model.NewIntVar(0, max_length_scaled, f"y_{i}")
        y_vars.append(y)
    
    fabric_length = model.NewIntVar(0, max_length_scaled, 'fabric_length')
    
    n = len(all_pieces)
    
    # Constraints
    # 1. Pieces must fit within fabric width
    for i, piece in enumerate(all_pieces):
        width_scaled = int(piece['width'] * SCALE)
        model.Add(x_vars[i] + width_scaled <= fabric_width_scaled)
    
    # 2. Fabric length must accommodate all pieces
    for i, piece in enumerate(all_pieces):
        length_scaled = int(piece['length'] * SCALE)
        model.Add(y_vars[i] + length_scaled <= fabric_length)
    
    # 3. Non-overlapping constraints
    for i in range(n):
        for j in range(i + 1, n):
            width_i = int(all_pieces[i]['width'] * SCALE)
            length_i = int(all_pieces[i]['length'] * SCALE)
            width_j = int(all_pieces[j]['width'] * SCALE)
            length_j = int(all_pieces[j]['length'] * SCALE)
            
            left = model.NewBoolVar(f'left_{i}_{j}')
            right = model.NewBoolVar(f'right_{i}_{j}')
            below = model.NewBoolVar(f'below_{i}_{j}')
            above = model.NewBoolVar(f'above_{i}_{j}')
            
            model.AddBoolOr([left, right, below, above])
            
            model.Add(x_vars[i] + width_i <= x_vars[j]).OnlyEnforceIf(left)
            model.Add(x_vars[j] + width_j <= x_vars[i]).OnlyEnforceIf(right)
            model.Add(y_vars[i] + length_i <= y_vars[j]).OnlyEnforceIf(below)
            model.Add(y_vars[j] + length_j <= y_vars[i]).OnlyEnforceIf(above)
    
    # Objective: Minimize fabric length
    model.Minimize(fabric_length)
    
    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.log_search_progress = False
    
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        optimal_length = solver.Value(fabric_length) / SCALE
        
        result_pieces = []
        for i, piece in enumerate(all_pieces):
            result_pieces.append({
                'name': piece['name'],
                'length': piece['length'],
                'width': piece['width'],
                'x': solver.Value(x_vars[i]) / SCALE,
                'y': solver.Value(y_vars[i]) / SCALE
            })
        
        total_pieces_area = sum(p['length'] * p['width'] for p in all_pieces)
        efficiency = (total_pieces_area / (optimal_length * fabric_width)) * 100
        
        return {
            'success': True,
            'status': 'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE',
            'fabric_length': optimal_length,
            'fabric_width': fabric_width,
            'efficiency': efficiency,
            'waste': 100 - efficiency,
            'waste_area': (optimal_length * fabric_width) - total_pieces_area,
            'solve_time': solver.WallTime(),
            'pieces': result_pieces
        }
    else:
        return {'success': False, 'error': 'No solution found'}


def draw_layout(result):
    """Draw the fabric layout using matplotlib"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw fabric background
    fabric_rect = patches.Rectangle(
        (0, 0),
        result['fabric_width'],
        result['fabric_length'],
        linewidth=2,
        edgecolor='black',
        facecolor='#f8f9fa',
        alpha=0.3
    )
    ax.add_patch(fabric_rect)
    
    # Draw grid
    for i in range(0, int(result['fabric_width']) + 1, 10):
        ax.axvline(x=i, color='gray', linewidth=0.3, alpha=0.5, linestyle='--')
    for i in range(0, int(result['fabric_length']) + 1, 10):
        ax.axhline(y=i, color='gray', linewidth=0.3, alpha=0.5, linestyle='--')
    
    # Draw pieces
    import random
    random.seed(42)
    
    for idx, piece in enumerate(result['pieces']):
        # Generate color
        hue = (idx * 137.5) % 360
        r = int((1 + (hue % 120) / 120) * 127)
        g = int((1 + ((hue + 120) % 120) / 120) * 127)
        b = int((1 + ((hue + 240) % 120) / 120) * 127)
        color = f'#{r:02x}{g:02x}{b:02x}'
        
        rect = patches.Rectangle(
            (piece['x'], piece['y']),
            piece['width'],
            piece['length'],
            linewidth=2,
            edgecolor='#2c3e50',
            facecolor=color,
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add label
        center_x = piece['x'] + piece['width'] / 2
        center_y = piece['y'] + piece['length'] / 2
        
        label = f"{piece['name']}\n{piece['width']:.0f}×{piece['length']:.0f}"
        ax.text(
            center_x, center_y,
            label,
            ha='center', va='center',
            fontsize=9,
            fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.6)
        )
    
    ax.set_xlim(-5, result['fabric_width'] + 5)
    ax.set_ylim(-5, result['fabric_length'] + 5)
    ax.set_aspect('equal')
    ax.set_xlabel('Width (cm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Length (cm)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Optimized Fabric Layout - {result["status"]}\n'
        f'Efficiency: {result["efficiency"]:.1f}% | Waste: {result["waste"]:.1f}%',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Header
st.markdown('<div class="main-header">✂️ Fabric Cutting Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Minimize waste, maximize efficiency with optimal fabric layout</div>', unsafe_allow_html=True)

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    fabric_width = st.number_input(
        "Fabric Width (cm)",
        min_value=1.0,
        value=150.0,
        step=0.1,
        help="The width of your fabric roll"
    )
    
    fabric_price = st.number_input(
        "Price per Meter (optional)",
        min_value=0.0,
        value=10.0,
        step=0.01,
        help="Cost per meter for cost calculation"
    )
    
    time_limit = st.slider(
        "Optimization Time (seconds)",
        min_value=10,
        max_value=300,
        value=60,
        step=10,
        help="Longer time = potentially better results"
    )
    
    st.markdown("---")
    
    st.header("📦 Pieces")
    
    # Input method selection
    input_method = st.radio(
        "Input Method",
        ["Manual Entry", "Upload CSV"],
        help="Choose how to input your pieces"
    )
    
    if input_method == "Manual Entry":
        st.markdown("**Add/Edit Pieces:**")
        
        # Display existing pieces
        for i, piece in enumerate(st.session_state.pieces):
            with st.expander(f"Piece {i+1}: {piece['name']}", expanded=False):
                col1, col2 = st.columns(2)
                
                piece['name'] = st.text_input(
                    "Name",
                    value=piece['name'],
                    key=f"name_{i}"
                )
                
                with col1:
                    piece['length'] = st.number_input(
                        "Length (cm)",
                        min_value=0.1,
                        value=float(piece['length']),
                        step=0.1,
                        key=f"length_{i}"
                    )
                    piece['width'] = st.number_input(
                        "Width (cm)",
                        min_value=0.1,
                        value=float(piece['width']),
                        step=0.1,
                        key=f"width_{i}"
                    )
                
                with col2:
                    piece['quantity'] = st.number_input(
                        "Quantity",
                        min_value=1,
                        value=int(piece['quantity']),
                        step=1,
                        key=f"quantity_{i}"
                    )
                    
                    if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                        st.session_state.pieces.pop(i)
                        st.rerun()
        
        if st.button("➕ Add New Piece", use_container_width=True):
            st.session_state.pieces.append({
                'name': f'Piece_{len(st.session_state.pieces)+1}',
                'length': 50.0,
                'width': 40.0,
                'quantity': 1
            })
            st.rerun()
    
    else:  # CSV Upload
        st.markdown("**Upload CSV with columns:**")
        st.code("name,length,width,quantity")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ['name', 'length', 'width', 'quantity']
                
                if all(col in df.columns for col in required_cols):
                    st.session_state.pieces = df.to_dict('records')
                    st.success(f"✅ Loaded {len(st.session_state.pieces)} pieces!")
                else:
                    st.error(f"❌ CSV must have columns: {', '.join(required_cols)}")
            except Exception as e:
                st.error(f"❌ Error reading CSV: {e}")
    
    st.markdown("---")
    
    # Optimize button
    if st.button("🚀 Optimize Layout", use_container_width=True, type="primary"):
        if len(st.session_state.pieces) == 0:
            st.error("❌ Please add at least one piece!")
        else:
            with st.spinner("Optimizing... This may take up to " + str(time_limit) + " seconds"):
                result = optimize_fabric(
                    st.session_state.pieces,
                    fabric_width,
                    time_limit
                )
                
                if result and result['success']:
                    st.session_state.solution = result
                    st.session_state.solution['fabric_price'] = fabric_price
                    st.success("✅ Optimization complete!")
                    st.rerun()
                elif result:
                    st.error(f"❌ {result.get('error', 'Optimization failed')}")
                    st.session_state.solution = None

# Main content
if st.session_state.solution is None:
    # Welcome screen
    st.info("""
    ### 👋 Welcome to Fabric Cutting Optimizer!
    
    **How to use:**
    1. Configure fabric width and price in the sidebar
    2. Add your pieces (length, width, quantity)
    3. Click "Optimize Layout"
    4. Get the most efficient cutting plan!
    
    **Features:**
    - ✅ Optimal fabric layout using Google OR-Tools
    - ✅ Minimize waste and maximize efficiency
    - ✅ Visual layout with exact coordinates
    - ✅ Cost calculation
    - ✅ Export cutting plan
    """)
    
    # Show input summary
    if len(st.session_state.pieces) > 0:
        st.subheader("📋 Current Pieces")
        
        df = pd.DataFrame(st.session_state.pieces)
        df['Area (cm²)'] = df['length'] * df['width'] * df['quantity']
        
        st.dataframe(df, use_container_width=True)
        
        total_area = df['Area (cm²)'].sum()
        total_pieces = df['quantity'].sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Pieces", int(total_pieces))
        with col2:
            st.metric("Total Area", f"{total_area:.1f} cm²")

else:
    # Display results
    result = st.session_state.solution
    
    # Metrics
    st.subheader("📊 Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Fabric Length",
            f"{result['fabric_length']:.2f} cm",
            help="Total fabric length needed"
        )
    
    with col2:
        st.metric(
            "Efficiency",
            f"{result['efficiency']:.1f}%",
            help="Percentage of fabric used (not wasted)"
        )
    
    with col3:
        st.metric(
            "Waste",
            f"{result['waste']:.1f}%",
            delta=f"-{result['waste']:.1f}%",
            delta_color="inverse",
            help="Percentage of fabric wasted"
        )
    
    with col4:
        if result['fabric_price'] > 0:
            total_cost = (result['fabric_length'] / 100) * result['fabric_price']
            st.metric(
                "Total Cost",
                f"${total_cost:.2f}",
                help="Total fabric cost"
            )
        else:
            st.metric(
                "Status",
                result['status'],
                help="Optimization status"
            )
    
    # Additional details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Fabric Width:** {result['fabric_width']:.1f} cm")
    with col2:
        st.info(f"**Total Area:** {result['fabric_length'] * result['fabric_width']:.1f} cm²")
    with col3:
        st.info(f"**Solve Time:** {result['solve_time']:.2f} seconds")
    
    st.markdown("---")
    
    # Visualization
    st.subheader("🎨 Fabric Layout")
    
    fig = draw_layout(result)
    st.pyplot(fig)
    
    # Download button for image
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    st.download_button(
        label="📥 Download Layout Image",
        data=buf,
        file_name="fabric_layout.png",
        mime="image/png",
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Cutting plan
    st.subheader("✂️ Cutting Plan")
    
    df_cutting = pd.DataFrame(result['pieces'])
    df_cutting = df_cutting[['name', 'width', 'length', 'x', 'y']]
    df_cutting.columns = ['Piece Name', 'Width (cm)', 'Length (cm)', 'X Position', 'Y Position']
    df_cutting.index = range(1, len(df_cutting) + 1)
    
    st.dataframe(df_cutting, use_container_width=True)
    
    # Download CSV
    csv = df_cutting.to_csv(index=True)
    st.download_button(
        label="📥 Download Cutting Plan (CSV)",
        data=csv,
        file_name="cutting_plan.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Instructions
    with st.expander("📋 How to Use This Cutting Plan"):
        st.markdown("""
        **Instructions for cutting:**
        
        1. Mark your fabric with X=0 (left edge) and Y=0 (bottom edge) as reference points
        2. Measure and mark each piece's position from these reference points
        3. Use the Width and Length columns to mark the piece dimensions
        4. Cut pieces according to the marked positions
        
        **Tips:**
        - Print this cutting plan for your cutting team
        - Use the visual layout above as a reference guide
        - Double-check measurements before cutting
        """)
