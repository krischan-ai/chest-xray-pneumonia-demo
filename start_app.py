import streamlit.web.bootstrap

# 直接运行app.py，绕过邮箱输入
streamlit.web.bootstrap.run(
    "app.py",
    is_hello=False,
    args=[],
    flag_options={}
)