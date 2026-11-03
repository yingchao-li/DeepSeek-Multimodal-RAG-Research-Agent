from ddgs import DDGS


def main():
    query = input("请输入搜索内容: ")

    print(f"\n🔍 正在搜索: {query}\n")

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)

        for i, r in enumerate(results, start=1):
            print(f"结果 {i}")
            print(f"标题: {r.get('title')}")
            print(f"链接: {r.get('href')}")
            print(f"摘要: {r.get('body')}")
            print("-" * 50)


if __name__ == "__main__":
    main()

