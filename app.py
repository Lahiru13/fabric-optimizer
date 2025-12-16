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

if 'unit' not in st.session_state:
    st.session_state.unit = 'cm'

# Unit conversion factors (all to cm)
UNIT_CONVERSIONS = {
    'cm': 1.0,
    'inch': 2.54,
    'meter': 100.0,
    'yard': 91.44
}

def convert_to_cm(value, unit):
    """Convert value from given unit to cm"""
    return value * UNIT_CONVERSIONS[unit]

def convert_from_cm(value, unit):
    """Convert value from cm to given unit"""
    return value / UNIT_CONVERSIONS[unit]

def get_unit_label(unit):
    """Get display label for unit"""
    labels = {
        'cm': 'cm',
        'inch': 'inches',
        'meter': 'meters',
        'yard': 'yards'
    }
    return labels[unit]


def optimize_fabric(pieces, fabric_width, time_limit):
    """Optimize fabric layout using OR-Tools CP-SAT solver"""
    
    # Expand pieces based on quantity
    all_pieces = []
    for p in pieces:
        for i in range(int(p['quantity'])):
            all_pieces.append({
                'width': float(p['length']),  # Swap for horizontal layout
                'length': float(p['width']),
                'original_length': float(p['length']),
                'original_width': float(p['width']),
                'name': f"{p['name']}_{i+1}" if p['quantity'] > 1 else p['name']
            })
    
    if not all_pieces:
        return None
    
    # Check if any piece exceeds fabric width
    for piece in all_pieces:
        if piece['length'] > fabric_width:
            st.error(f"❌ Error: {piece['name']} width ({piece['original_width']}cm) exceeds fabric width ({fabric_width}cm)!")
            return None
    
    # Estimate max length
    total_area = sum(p['width'] * p['length'] for p in all_pieces)
    theoretical_min = total_area / fabric_width
    max_length = int(theoretical_min * 2.0)
    
    # Create model
    model = cp_model.CpModel()
    SCALE = 100
    
    fabric_width_scaled = int(fabric_width * SCALE)
    max_length_scaled = int(max_length * SCALE)
    
    # Variables
    x_vars = []
    y_vars = []
    
    for i, piece in enumerate(all_pieces):
        x = model.NewIntVar(0, max_length_scaled, f"x_{i}")
        x_vars.append(x)
        y = model.NewIntVar(0, fabric_width_scaled, f"y_{i}")
        y_vars.append(y)
    
    fabric_length = model.NewIntVar(0, max_length_scaled, 'fabric_length')
    
    n = len(all_pieces)
    
    # Constraints
    # 1. Pieces must fit within fabric width (Y-axis)
    for i, piece in enumerate(all_pieces):
        length_scaled = int(piece['length'] * SCALE)
        model.Add(y_vars[i] + length_scaled <= fabric_width_scaled)
    
    # 2. Fabric length (X-axis) must accommodate all pieces
    for i, piece in enumerate(all_pieces):
        width_scaled = int(piece['width'] * SCALE)
        model.Add(x_vars[i] + width_scaled <= fabric_length)
    
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
                'length': piece['original_length'],
                'width': piece['original_width'],
                'x': solver.Value(x_vars[i]) / SCALE,
                'y': solver.Value(y_vars[i]) / SCALE
            })
        
        total_pieces_area = sum(p['width'] * p['length'] for p in all_pieces)
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
        return {'success': False, 'error': f'No solution found within {time_limit} seconds. Try increasing the time limit.'}


def draw_layout(result, unit='cm'):
    """Draw the fabric layout using matplotlib"""
    
    # Convert values for display
    fabric_length_display = convert_from_cm(result['fabric_length'], unit)
    fabric_width_display = convert_from_cm(result['fabric_width'], unit)
    unit_label = get_unit_label(unit)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Draw fabric background
    fabric_rect = patches.Rectangle(
        (0, 0),
        fabric_length_display,
        fabric_width_display,
        linewidth=2,
        edgecolor='black',
        facecolor='#f8f9fa',
        alpha=0.3
    )
    ax.add_patch(fabric_rect)
    
    # Calculate grid spacing based on unit
    if unit == 'cm':
        grid_spacing = 10
    elif unit == 'inch':
        grid_spacing = 2
    elif unit == 'meter':
        grid_spacing = 0.1
    else:  # yard
        grid_spacing = 0.25
    
    # Draw grid with measurements
    x_max = int(fabric_length_display / grid_spacing) + 1
    for i in range(x_max + 1):
        x_pos = i * grid_spacing
        if x_pos <= fabric_length_display:
            ax.axvline(x=x_pos, color='gray', linewidth=0.3, alpha=0.5, linestyle='--')
            if i > 0:
                ax.text(x_pos, -fabric_width_display * 0.03, f'{int(x_pos) if x_pos == int(x_pos) else x_pos:.1f}',
                       ha='center', va='top', fontsize=8, color='#666')
    
    y_max = int(fabric_width_display / grid_spacing) + 1
    for i in range(y_max + 1):
        y_pos = i * grid_spacing
        if y_pos <= fabric_width_display:
            ax.axhline(y=y_pos, color='gray', linewidth=0.3, alpha=0.5, linestyle='--')
            if i > 0:
                ax.text(-fabric_length_display * 0.015, y_pos, f'{int(y_pos) if y_pos == int(y_pos) else y_pos:.1f}',
                       ha='right', va='center', fontsize=8, color='#666')
    
    # Color palette
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B195', '#C06C84',
        '#6C5B7B', '#355C7D', '#F67280', '#C7CEEA', '#99B898',
        '#E84A5F', '#FF847C', '#FECEAB', '#A8E6CF', '#DCEDC1'
    ]
    
    # Draw pieces
    for idx, piece in enumerate(result['pieces']):
        color = colors[idx % len(colors)]
        
        x_display = convert_from_cm(piece['x'], unit)
        y_display = convert_from_cm(piece['y'], unit)
        length_display = convert_from_cm(piece['length'], unit)
        width_display = convert_from_cm(piece['width'], unit)
        
        rect = patches.Rectangle(
            (x_display, y_display),
            length_display,
            width_display,
            linewidth=2,
            edgecolor='#2c3e50',
            facecolor=color,
            alpha=0.75
        )
        ax.add_patch(rect)
        
        # Label
        center_x = x_display + length_display / 2
        center_y = y_display + width_display / 2
        
        min_dimension = min(length_display, width_display)
        
        if unit == 'cm':
            if min_dimension < 20:
                fontsize = 5
            elif min_dimension < 30:
                fontsize = 6
            elif min_dimension < 40:
                fontsize = 7
            else:
                fontsize = 8
        else:
            fontsize = 7
        
        label = f"{piece['name']}\n{length_display:.1f}×{width_display:.1f}"
        
        if min_dimension >= (15 if unit == 'cm' else 5):
            ax.text(
                center_x, center_y,
                label,
                ha='center', va='center',
                fontsize=fontsize,
                fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
            )
    
    ax.set_xlim(-fabric_length_display * 0.05, fabric_length_display * 1.02)
    ax.set_ylim(-fabric_width_display * 0.08, fabric_width_display * 1.02)
    ax.set_aspect('equal')
    ax.set_xlabel(f'Fabric Length ({unit_label})', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Fabric Width ({unit_label})', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Optimized Fabric Layout - {result["status"]}\n'
        f'Length: {fabric_length_display:.2f}{unit_label} × Width: {fabric_width_display:.1f}{unit_label} | '
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
    
    # Unit selection
    unit = st.selectbox(
        "Measurement Unit",
        options=['cm', 'inch', 'meter', 'yard'],
        index=0,
        format_func=lambda x: {'cm': 'Centimeters (cm)', 'inch': 'Inches', 'meter': 'Meters', 'yard': 'Yards'}[x],
        help="Choose your preferred measurement unit"
    )
    st.session_state.unit = unit
    unit_label = get_unit_label(unit)
    
    fabric_width_input = st.number_input(
        f"Fabric Width ({unit_label})",
        min_value=0.1,
        value=150.0 if unit == 'cm' else convert_from_cm(150.0, unit),
        step=0.1,
        help="The width of your fabric roll"
    )
    fabric_width = convert_to_cm(fabric_width_input, unit)
    
    fabric_price = st.number_input(
        f"Price per {unit_label} (optional)",
        min_value=0.0,
        value=10.0 if unit in ['cm', 'meter'] else 5.0,
        step=0.01,
        help="Cost per unit for cost calculation"
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
                        f"Length ({unit_label})",
                        min_value=0.1,
                        value=convert_from_cm(float(piece['length']), unit) if 'converted' not in piece else float(piece['length']),
                        step=0.1,
                        key=f"length_{i}"
                    )
                    piece['width'] = st.number_input(
                        f"Width ({unit_label})",
                        min_value=0.1,
                        value=convert_from_cm(float(piece['width']), unit) if 'converted' not in piece else float(piece['width']),
                        step=0.1,
                        key=f"width_{i}"
                    )
                    piece['converted'] = True
                
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
            # Convert all piece dimensions to cm for optimization
            pieces_in_cm = []
            for piece in st.session_state.pieces:
                pieces_in_cm.append({
                    'name': piece['name'],
                    'length': convert_to_cm(float(piece['length']), unit),
                    'width': convert_to_cm(float(piece['width']), unit),
                    'quantity': piece['quantity']
                })
            
            with st.spinner("Optimizing... This may take up to " + str(time_limit) + " seconds"):
                result = optimize_fabric(
                    pieces_in_cm,
                    fabric_width,
                    time_limit
                )
                
                if result and result['success']:
                    st.session_state.solution = result
                    st.session_state.solution['fabric_price'] = fabric_price
                    st.session_state.solution['unit'] = unit
                    st.success("✅ Optimization complete!")
                    st.rerun()
                elif result:
                    st.error(f"❌ {result.get('error', 'Optimization failed')}")
                    st.session_state.solution = None

# Main content
if st.session_state.solution is None:
    st.info("""
    ### 👋 Welcome to Fabric Cutting Optimizer!
    
    **How to use:**
    1. Select your measurement unit (cm/inch/meter/yard)
    2. Configure fabric width and price in the sidebar
    3. Add your pieces (length, width, quantity)
    4. Click "Optimize Layout"
    5. Get the most efficient cutting plan!
    
    **Features:**
    - ✅ Optimal fabric layout using Google OR-Tools
    - ✅ Multiple unit support (cm, inches, meters, yards)
    - ✅ Minimize waste and maximize efficiency
    - ✅ Visual layout with measurements
    - ✅ Cost calculation
    - ✅ Export cutting plan
    """)
    
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
    display_unit = result.get('unit', 'cm')
    unit_label = get_unit_label(display_unit)
    
    fabric_length_display = convert_from_cm(result['fabric_length'], display_unit)
    fabric_width_display = convert_from_cm(result['fabric_width'], display_unit)
    
    st.subheader("📊 Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            f"Fabric Length",
            f"{fabric_length_display:.2f} {unit_label}",
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
            total_cost = fabric_length_display * result['fabric_price']
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
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Fabric Width:** {fabric_width_display:.1f} {unit_label}")
    with col2:
        st.info(f"**Total Area:** {(fabric_length_display * fabric_width_display):.1f} {unit_label}²")
    with col3:
        st.info(f"**Solve Time:** {result['solve_time']:.2f} seconds")
    
    st.markdown("---")
    
    st.subheader("🎨 Fabric Layout")
    
    fig = draw_layout(result, display_unit)
    st.pyplot(fig)
    
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
    
    st.subheader("✂️ Cutting Plan")
    
    df_cutting = pd.DataFrame(result['pieces'])
    df_cutting = df_cutting[['name', 'width', 'length', 'x', 'y']]
    df_cutting.columns = ['Piece Name', f'Width ({unit_label})', f'Length ({unit_label})', 'X Position', 'Y Position']
    
    # Convert to display unit
    for col in [f'Width ({unit_label})', f'Length ({unit_label})', 'X Position', 'Y Position']:
        if col in df_cutting.columns:
            df_cutting[col] = df_cutting[col].apply(lambda x: convert_from_cm(x, display_unit))
    
    df_cutting.index = range(1, len(df_cutting) + 1)
    
    st.dataframe(df_cutting, use_container_width=True)
    
    csv = df_cutting.to_csv(index=True)
    st.download_button(
        label="📥 Download Cutting Plan (CSV)",
        data=csv,
        file_name="cutting_plan.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    with st.expander("📋 How to Use This Cutting Plan"):
        st.markdown(f"""
        **Instructions for cutting:**
        
        1. Mark your fabric with X=0 (left edge) and Y=0 (bottom edge) as reference points
        2. All measurements are in {unit_label}
        3. Measure and mark each piece's position from these reference points
        4. Cut pieces according to the marked positions
        
        **Tips:**
        - Print this cutting plan for your cutting team
        - Use the visual layout above as a reference guide
        - Double-check measurements before cutting
        """)
