import streamlit as st
import pickle
import pandas as pd
import numpy as np # 虽然模型加载不直接用，但pandas和sklearn底层可能依赖，包含更稳妥

# 设置页面标题和图标
st.set_page_config(page_title="动漫总播放量预测器", page_icon="📺")

# --- 模型加载 ---
# 定义模型文件名
MODEL_FILENAME = 'random_forest_model.pkl'

@st.cache_resource # 使用st.cache_resource缓存模型加载，避免每次用户交互都重新加载
def load_model(filename):
    """加载训练好的模型"""
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        st.success("✅ 模型加载成功！")
        return model
    except FileNotFoundError:
        st.error(f"❌ 错误: 模型文件 '{filename}' 未找到。请确保模型文件与app.py在同一目录下。")
        return None
    except Exception as e:
        st.error(f"❌ 加载模型时出错: {str(e)}")
        return None

# 加载模型
model = load_model(MODEL_FILENAME)

# 定义模型期望的特征顺序
# 这与你原始代码中定义的 features 列表一致
required_features = ['类型', '是否改编', '开播时间', '是否独家', '产地', '集数', '点赞数（PV）',
                   '投币数（PV）', '收藏数（PV）', '分享数（PV）', 'Topic 0', 'Topic 1',
                   'Topic 2', 'Topic 3', 'Topic 4']

# 定义类别特征的映射，方便用户选择时显示文本
type_options = {
    1: '少儿教育',
    2: '幻想冒险',
    3: '情感生活',
    4: '悬疑惊悚',
    5: '文艺历史'
}

adapted_options = {
    0: '否',
    1: '是'
}

air_time_options = {
    0: '非假期档 (其他月份)',
    1: '假期档 (1,2,7,8月)'
}

exclusive_options = {
    0: '否',
    1: '是'
}

origin_options = {
    1: '日本',
    2: '美国',
    3: '中国'
}


# --- 页面布局和输入控件 ---
st.title("📊 动漫总播放量预测应用")

st.write("""
欢迎使用动漫总播放量预测应用。请在下方输入动漫的相关特征，我们将使用预训练的随机森林模型为您预测其预计的总播放量（万）。

您的模型文件 `random_forest_model.pkl` 已加载。
""")

if model is not None:
    st.header("请填写动漫特征")

    # 使用列布局使输入更整洁
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("基本信息")
        # 类别型/二值型特征使用 selectbox
        type_val = st.selectbox(
            "类型",
            options=list(type_options.keys()),
            format_func=lambda x: type_options[x],
            help="选择动漫的主要类型"
        )
        adapted_val = st.selectbox(
            "是否改编",
            options=list(adapted_options.keys()),
            format_func=lambda x: adapted_options[x],
            help="该动漫是否由漫画、小说等改编而来"
        )
        air_time_val = st.selectbox(
            "开播时间",
            options=list(air_time_options.keys()),
            format_func=lambda x: air_time_options[x],
            help="动漫的开播时间是否在假期档（1, 2, 7, 8月）"
        )
        exclusive_val = st.selectbox(
            "是否独家",
            options=list(exclusive_options.keys()),
            format_func=lambda x: exclusive_options[x],
            help="该动漫是否为平台独家播放"
        )
        origin_val = st.selectbox(
            "产地",
            options=list(origin_options.keys()),
            format_func=lambda x: origin_options[x],
            help="动漫的主要制作国家或地区"
        )

    with col2:
        st.subheader("数据指标")
        # 数值型特征使用 number_input
        episodes_val = st.number_input("集数", min_value=1, max_value=500, value=12, help="动漫的总集数")
        likes_val = st.number_input("点赞数（个）", min_value=0, value=50000, help="该动漫的点赞数量（PV）")
        coins_val = st.number_input("投币数（个）", min_value=0, value=20000, help="该动漫的投币数量（PV）")
        collects_val = st.number_input("收藏数（个）", min_value=0, value=10000, help="该动漫的收藏数量（PV）")
        shares_val = st.number_input("分享数（个）", min_value=0, value=5000, help="该动漫的分享数量（PV）")

    with col3:
        st.subheader("主题权重 (0-1之间)")
        # Topic特征使用 slider
        topic_0_val = st.slider("Topic 0 权重", min_value=0.0, max_value=1.0, value=0.1, step=0.01, help="该动漫在主题Topic 0上的权重得分")
        topic_1_val = st.slider("Topic 1 权重", min_value=0.0, max_value=1.0, value=0.1, step=0.01, help="该动漫在主题Topic 1上的权重得分")
        topic_2_val = st.slider("Topic 2 权重", min_value=0.0, max_value=1.0, value=0.1, step=0.01, help="该动漫在主题Topic 2上的权重得分")
        topic_3_val = st.slider("Topic 3 权重", min_value=0.0, max_value=1.0, value=0.1, step=0.01, help="该动漫在主题Topic 3上的权重得分")
        topic_4_val = st.slider("Topic 4 权重", min_value=0.0, max_value=1.0, value=0.1, step=0.01, help="该动漫在主题Topic 4上的权重得分")

    # --- 预测按钮和结果显示 ---
    st.markdown("---") # 添加分隔线
    if st.button("点击预测总播放量"):
        if model is not None:
            # 收集所有输入值到一个字典
            input_data = {
                '类型': type_val,
                '是否改编': adapted_val,
                '开播时间': air_time_val,
                '是否独家': exclusive_val,
                '产地': origin_val,
                '集数': episodes_val,
                '点赞数（PV）': likes_val,
                '投币数（PV）': coins_val,
                '收藏数（PV）': collects_val,
                '分享数（PV）': shares_val,
                'Topic 0': topic_0_val,
                'Topic 1': topic_1_val,
                'Topic 2': topic_2_val,
                'Topic 3': topic_3_val,
                'Topic 4': topic_4_val
            }

            # 转换为DataFrame并确保列顺序正确
            input_df = pd.DataFrame([input_data])
            input_df = input_df[required_features] # 按照模型训练时的特征顺序排列

            # 进行预测
            try:
                prediction = model.predict(input_df)
                # 显示预测结果
                st.balloons() # 添加预测成功的动画效果
                st.success(f"🎉 预测的总播放量为: **{prediction[0]:,.2f} 万**")
                st.write(f"（预测结果基于您输入的特征值和随机森林模型计算得出）")

            except Exception as e:
                st.error(f"❌ 预测过程中发生错误: {str(e)}")
        else:
            st.warning("模型尚未成功加载，无法进行预测。请检查模型文件。")

# 可选：添加一些关于特征的解释或应用说明
st.markdown("---")
st.markdown(
    """
    **特征说明:**
    *   **类型, 产地:** 经过预处理的类别编码。
    *   **是否改编, 开播时间, 是否独家:** 二值特征，通常0表示否，1表示是（开播时间1表示假期档，0表示非假期档）。
    *   **集数:** 动漫的总集数。
    *   **点赞/投币/收藏/分享数（PV）:** 表示该动漫在平台上的用户互动数据，通常是经过一定处理的累计值。
    *   **Topic 0-4:** 可能是通过主题模型（如LDA）从动漫简介或标签中提取出的主题分布权重，表示该动漫内容在不同主题上的倾向性。
    """
)

# 可选：添加页脚
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: gray;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    <div class="footer">
        随机森林动漫总播放量预测应用 by [你的名字/团队名]
    </div>
    """,
    unsafe_allow_html=True
)