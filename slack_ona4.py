import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from slack_sdk import WebClient
from datetime import datetime, timedelta
import plotly.graph_objects as go
from collections import defaultdict
import numpy as np

def initialize_slack_client(token):
    """Initialize Slack client with provided token"""
    return WebClient(token=token)

def get_channels(client):
    """Fetch all channels and return as a dictionary"""
    channels = []
    try:
        result = client.conversations_list(types="public_channel,private_channel")
        channels.extend(result['channels'])
        
        while result.get('response_metadata', {}).get('next_cursor'):
            cursor = result['response_metadata']['next_cursor']
            result = client.conversations_list(
                types="public_channel,private_channel",
                cursor=cursor
            )
            channels.extend(result['channels'])
        
        # Create a dictionary with channel names as keys and IDs as values
        channel_dict = {
            f"#{channel['name']}": channel['id'] 
            for channel in channels 
            if not channel['is_archived']
        }
        return channel_dict
    except Exception as e:
        st.error(f"Error fetching channels: {str(e)}")
        return {}

def get_user_info(client, user_id):
    """Get user information from user ID"""
    try:
        result = client.users_info(user=user_id)
        user = result['user']
        return user.get('real_name', user.get('name', user_id))
    except Exception:
        return user_id

def fetch_channel_messages(client, channel_id, days_back=30):
    """Fetch messages from a specific channel within time range"""
    messages = []
    oldest = datetime.now() - timedelta(days=days_back)
    oldest_timestamp = oldest.timestamp()
    
    try:
        result = client.conversations_history(
            channel=channel_id,
            oldest=oldest_timestamp
        )
        messages.extend(result['messages'])
        
        while result.get('response_metadata', {}).get('next_cursor'):
            cursor = result['response_metadata']['next_cursor']
            result = client.conversations_history(
                channel=channel_id,
                cursor=cursor
            )
            messages.extend(result['messages'])
            
    except Exception as e:
        st.error(f"Error fetching messages: {str(e)}")
    
    return messages

def build_interaction_network(client, messages, channel_id):
    """Build network graph from messages"""
    G = nx.Graph()
    interactions = defaultdict(int)
    user_names = {}
    
    for msg in messages:
        sender = msg.get('user')
        if not sender:
            continue
            
        # Get sender's name if not already cached
        if sender not in user_names:
            user_names[sender] = get_user_info(client, sender)
            
        # Count replies
        if 'thread_ts' in msg:
            try:
                thread_messages = client.conversations_replies(
                    channel=channel_id,
                    ts=msg['thread_ts']
                )['messages']
                
                for reply in thread_messages:
                    reply_user = reply.get('user')
                    if reply_user and reply_user != sender:
                        if reply_user not in user_names:
                            user_names[reply_user] = get_user_info(client, reply_user)
                        pair = tuple(sorted([user_names[sender], user_names[reply_user]]))
                        interactions[pair] += 1
            except Exception as e:
                st.warning(f"Could not fetch some thread replies: {str(e)}")
        
        # Count mentions
        if 'text' in msg:
            mentions = [word[2:-1] for word in msg['text'].split() if word.startswith('<@') and word.endswith('>')]
            for mentioned_user in mentions:
                if mentioned_user not in user_names:
                    user_names[mentioned_user] = get_user_info(client, mentioned_user)
                pair = tuple(sorted([user_names[sender], user_names[mentioned_user]]))
                interactions[pair] += 1
    
    # Add edges to network
    for (user1, user2), weight in interactions.items():
        G.add_edge(user1, user2, weight=weight)
    
    return G

def calculate_network_metrics(G):
    """Calculate various network metrics"""
    metrics = {
        'Degree Centrality / Communication Hubs': {
            'values': nx.degree_centrality(G),
            'description': "Spot your most connected employees—those who naturally bring people together. They’re collaboration hubs."
        },
        'Betweenness Centrality / Bridges': {
            'values': nx.betweenness_centrality(G),
            'description': "Identify the bridges in your organization—employees connecting teams. They prevent silos but may face burnout."
        },
        'Clustering Coefficient / Cohesion': {
            'values': nx.clustering(G),
            'description': "Check if your teams are forming mini-communities. Strong clustering shows cohesive teamwork."
        },
        'PageRank/ Influence': {
            'values': nx.pagerank(G),
            'description': "Identify employees who amplify ideas and build momentum across teams. They hold organizational influence."
        }
    }
    return metrics

def calculate_daily_metrics(client, messages, channel_id):
    """Calculate network metrics for each day"""
    daily_metrics = defaultdict(lambda: defaultdict(list))
    
    # Group messages by day
    messages_by_day = defaultdict(list)
    for msg in messages:
        # Convert timestamp to date
        ts = float(msg.get('ts', 0))
        date = datetime.fromtimestamp(ts).date()
        messages_by_day[date].append(msg)
    
    # Calculate metrics for each day
    for date, daily_messages in messages_by_day.items():
        G = build_interaction_network(client, daily_messages, channel_id)
        if G.number_of_nodes() > 0:  # Only calculate if network has nodes
            metrics = {
                'Degree Centrality': nx.degree_centrality(G),
                'Betweenness Centrality': nx.betweenness_centrality(G),
                'Clustering Coefficient': nx.clustering(G),
                'Page Rank': nx.pagerank(G)
            }
            
            # Calculate average for each metric
            for metric_name, values in metrics.items():
                if values:  # Check if we have values
                    avg = sum(values.values()) / len(values)
                    daily_metrics[metric_name][date] = avg
    
    return daily_metrics

def plot_metric_trend(daily_metrics, metric_name, start_date=None, end_date=None):
    """Create a line chart for metric trends with date filtering"""
    dates = list(daily_metrics[metric_name].keys())
    values = list(daily_metrics[metric_name].values())
    
    # Convert to dataframe for easier date filtering
    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    # Apply date filters if provided
    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['value'],
        mode='lines+markers',
        name=metric_name,
        line=dict(width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f'{metric_name} Trend Over Time',
        xaxis_title='Date',
        yaxis_title='Average Value',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig


def plot_network(G):
    """Create interactive network visualization using Plotly"""
    pos = nx.spring_layout(G)
    
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Get node degrees for sizing and coloring
    degrees = dict(G.degree())
    
    # Create a more balanced scaling function using log scale
    min_degree = min(degrees.values()) if degrees else 0
    max_degree = max(degrees.values()) if degrees else 1
    
    def scale_size(d):
        # Use log scaling to prevent giant nodes
        if max_degree == min_degree:
            return 15
        base_size = 10
        max_size = 30
        if d == 0:
            return base_size
        # Log scaling with a minimum size
        scaled = base_size + (max_size - base_size) * (np.log(d + 1) / np.log(max_degree + 1))
        return scaled

    node_trace = go.Scatter(
        x=[], y=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            colorscale='Viridis',  # Using a colorful scale
            reversescale=True,
            size=[],
            color=[],  # Will be filled with degree values
            colorbar=dict(
                thickness=15,
                title='Number of Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=0.5, color='#fff')
        ))

    # Add edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Add nodes
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = degrees[node]
        node_colors.append(degree)  # Use degree for color
        node_sizes.append(scale_size(degree))
        node_text.append(f'User: {node}<br>Connections: {degree}')
        
    node_trace['x'] = node_x
    node_trace['y'] = node_y
    node_trace['text'] = node_text
    node_trace.marker['size'] = node_sizes
    node_trace.marker['color'] = node_colors

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        ))
    
    return fig

def calculate_org_metrics(G):
    """Calculate organization-level network metrics"""
    
    # Basic network statistics
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    density = nx.density(G)
    
    # Find isolates (nodes with no connections)
    #isolates = list(nx.isolates(G))
    
    # Calculate average clustering coefficient (local clustering)
    avg_clustering = nx.average_clustering(G)
    
    # Identify components (potential silos)
    components = list(nx.connected_components(G))
    num_components = len(components)
    
    # Calculate size of largest component
    largest_component_size = len(max(components, key=len)) if components else 0
    largest_component_ratio = largest_component_size / total_nodes if total_nodes > 0 else 0
    
    # Calculate average path length (only for largest component to avoid inf)
    largest_component = G.subgraph(max(components, key=len)) if components else G
    try:
        avg_path_length = nx.average_shortest_path_length(largest_component)
    except:
        avg_path_length = 0
    
    num_isolates = G.number_of_nodes() - largest_component_size
    
    # Calculate reciprocity (for responses/mentions)
    num_bidirectional = sum(1 for u, v in G.edges() if G.has_edge(v, u))
    reciprocity = num_bidirectional / total_edges if total_edges > 0 else 0
    
    # Identify potential brokers (nodes with high betweenness)
    betweenness = nx.betweenness_centrality(G)
    num_brokers = sum(1 for v in betweenness.values() if v > np.mean(list(betweenness.values())))
    
    return {
        "Total Members": total_nodes,
        "Active Members": total_nodes - num_isolates,
        "Total Interactions": total_edges,
        "Network Density": {
            "value": density,
            "description": "Percentage of potential connections that actually exist. Higher values indicate a more interconnected network."
        },
        "Isolated Members": {
            "value": num_isolates,
            "description": "Number of members with no interactions."
        },
        "Network Fragmentation": {
            "value": num_components,
            "description": "Number of disconnected groups. More than 1 indicates potential silos."
        },
        "Largest Group Size": {
            "value": largest_component_ratio,
            "description": "Percentage of members in the largest connected group. Lower values suggest fragmentation."
        },
        "Collaboration Score": {
            "value": avg_clustering,
            "description": "Measures how often people who interact with the same person also interact with each other."
        },
        "Average Distance": {
            "value": avg_path_length,
            "description": "Average number of steps needed to reach any member from any other member."
        },
        "Reciprocity": {
            "value": reciprocity,
            "description": "Percentage of interactions that are mutual. Higher values indicate two-way communication."
        },
        "Key Connectors": {
            "value": num_brokers,
            "description": "Number of members who frequently connect different groups."
        }
    }

def display_summary_dashboard(metrics):
    """Display the organization metrics in a dashboard layout"""
    st.subheader("Organization Network Summary")
    
    # Top level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Members", metrics["Total Members"])
    with col2:
        st.metric("Active Members", metrics["Active Members"])
    with col3:
        st.metric("Total Interactions", metrics["Total Interactions"])
    with col4:
        st.metric("Key Connectors", metrics["Key Connectors"]["value"])
    
    # Create three columns for detailed metrics
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Network Structure
    with col1:
        st.markdown("### Network Structure")
        st.metric(
            "Network Density",
            f"{metrics['Network Density']['value']:.1%}",
            help=metrics['Network Density']['description']
        )
        st.metric(
            "Isolated Members",
            metrics['Isolated Members']['value'],
            help=metrics['Isolated Members']['description']
        )
        st.metric(
            "Network Groups",
            metrics['Network Fragmentation']['value'],
            help=metrics['Network Fragmentation']['description']
        )
    
    # Column 2: Collaboration Metrics
    with col2:
        st.markdown("### Collaboration Metrics")
        st.metric(
            "Collaboration Score",
            f"{metrics['Collaboration Score']['value']:.1%}",
            help=metrics['Collaboration Score']['description']
        )
        st.metric(
            "Largest Group",
            f"{metrics['Largest Group Size']['value']:.1%}",
            help=metrics['Largest Group Size']['description']
        )
        st.metric(
            "Average Distance",
            f"{metrics['Average Distance']['value']:.2f}",
            help=metrics['Average Distance']['description']
        )
    
    # Column 3: Network Health
    with col3:
        st.markdown("### Network Health")
        
        # Network Density Indicator
        density = metrics['Network Density']['value']
        if density < 0.1:
            density_health = "🔴 Low Connectivity"
        elif density < 0.3:
            density_health = "🟡 Moderate Connectivity"
        else:
            density_health = "🟢 High Connectivity"
        
        # Isolation Indicator
        isolation_ratio = metrics['Isolated Members']['value'] / metrics['Total Members']
        if isolation_ratio > 0.2:
            isolation_health = "🔴 High Isolation"
        elif isolation_ratio > 0.1:
            isolation_health = "🟡 Moderate Isolation"
        else:
            isolation_health = "🟢 Low Isolation"
        
        # Fragmentation Indicator
        fragment_ratio = 1 - metrics['Largest Group Size']['value']
        if fragment_ratio > 0.3:
            fragment_health = "🔴 High Fragmentation"
        elif fragment_ratio > 0.1:
            fragment_health = "🟡 Moderate Fragmentation"
        else:
            fragment_health = "🟢 Low Fragmentation"
        
        st.markdown(f"**Network Density:**  \n{density_health}")
        st.markdown(f"**Member Isolation:**  \n{isolation_health}")
        st.markdown(f"**Network Fragmentation:**  \n{fragment_health}")

# Streamlit UI
st.title("Slack Organizational Network Analysis")

# Sidebar configuration
st.sidebar.header("Configuration")
slack_token = st.sidebar.text_input("Enter Slack API Token", type="password")

if slack_token:
    try:
        # Initialize client
        client = initialize_slack_client(slack_token)
        
        # Fetch channels
        with st.spinner("Fetching channels..."):
            channels = get_channels(client)
            
        if channels:
            channel_options = ["Select a channel"] + list(channels.keys())
            selected_channel = st.sidebar.selectbox(
                "Select Channel",
                options=channel_options,
                index=0
            )
            
            if selected_channel != "Select a channel":
                channel_id = channels[selected_channel]
                days_back = st.sidebar.slider("Days to Analyze", 1, 90, 30)
                
                if st.sidebar.button("Analyze Channel"):
                    # Fetch messages and store in session state
                    with st.spinner("Fetching messages..."):
                        st.session_state.messages = fetch_channel_messages(client, channel_id, days_back)
                        st.success(f"Retrieved {len(st.session_state.messages)} messages")
                    
                    # Build network and calculate metrics
                    with st.spinner("Building network and calculating metrics..."):
                        st.session_state.network = build_interaction_network(client, st.session_state.messages, channel_id)
                        st.session_state.metrics = calculate_network_metrics(st.session_state.network)
                        st.success(f"Network built with {st.session_state.network.number_of_nodes()} users and {st.session_state.network.number_of_edges()} connections")
                        
                        # Calculate daily metrics
                        st.session_state.daily_metrics = calculate_daily_metrics(client, st.session_state.messages, channel_id)

                # Check if analysis has been run
                if hasattr(st.session_state, 'network') and st.session_state.network is not None:
                    # Display network visualization
                    st.subheader("Network Visualization")
                    fig = plot_network(st.session_state.network)
                    st.plotly_chart(fig)
                    
                    with st.spinner("Calculating organization metrics..."):
                            org_metrics = calculate_org_metrics(st.session_state.network)
                            display_summary_dashboard(org_metrics)
                    
                    # Display metrics section
                    st.subheader("Network Metrics")
                    
                    # Add custom CSS for pill-style selector
                    st.markdown("""
                        <style>
                        div[data-baseweb="select"] > div {
                            background-color: #f0f2f6;
                            border-radius: 20px;
                            border: none;
                            padding: 2px 15px;
                        }
                        div[data-baseweb="select"] > div:hover {
                            background-color: #e8e8e8;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    # Create pill selector for metrics
                    selected_metric = st.selectbox(
                        "Choose metric",
                        options=list(st.session_state.metrics.keys()),
                        key="metric_selector"
                    )

                    # Display selected metric information and trend
                    if selected_metric:
                        # Display metric description
                        st.info(st.session_state.metrics[selected_metric]['description'])
                        
                        # Display top users
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.write("Top 10 Users")
                            top_users = dict(sorted(
                                st.session_state.metrics[selected_metric]['values'].items(),
                                key=lambda x: x[1],
                                reverse=True
                            )[:10])
                            st.dataframe(pd.Series(top_users, name=selected_metric))
                        
                        with col2:
                            st.write("Trend Over Time")
                            
                            # Get the date range from the daily metrics
                            if st.session_state.daily_metrics[selected_metric]:
                                min_date = min(st.session_state.daily_metrics[selected_metric].keys())
                                max_date = max(st.session_state.daily_metrics[selected_metric].keys())
                                
                                # Add date range selectors
                                date_col1, date_col2 = st.columns(2)
                                with date_col1:
                                    start_date = st.date_input(
                                        "Start Date",
                                        value=min_date,
                                        min_value=min_date,
                                        max_value=max_date
                                    )
                                with date_col2:
                                    end_date = st.date_input(
                                        "End Date",
                                        value=max_date,
                                        min_value=min_date,
                                        max_value=max_date
                                    )
                                
                                # Plot trend with date range
                                trend_fig = plot_metric_trend(
                                    st.session_state.daily_metrics,
                                    selected_metric,
                                    start_date,
                                    end_date
                                )
                                st.plotly_chart(trend_fig, use_container_width=True)
                            else:
                                st.warning("No trend data available for the selected metric")

                    # Export options
                    st.subheader("Export Data")
                    if st.button("Export Network Data"):
                        network_data = nx.to_pandas_adjacency(st.session_state.network)
                        st.download_button(
                            label="Download Network Matrix",
                            data=network_data.to_csv(),
                            file_name="network_data.csv",
                            mime="text/csv"
                        )
                    
        else:
            st.sidebar.error("No channels found or insufficient permissions")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please enter your Slack API token in the sidebar to begin analysis.")