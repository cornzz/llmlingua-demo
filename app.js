() => {
    (() => {
        if (document.cookie.includes('session=')) return;
        const date = new Date(+new Date() + 10 * 365 * 24 * 60 * 60 * 1000);
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        document.cookie = `session=${Array(32).fill().map(() => chars.charAt(Math.floor(Math.random() * chars.length))).join('')}; expires=${date.toUTCString()}; path=/`;
    })();

    wordCountHandler = (event) => {
        const words = event.target.value.match(/\w+/g)?.length ?? 0;
        const label = event.target.previousElementSibling;
        label.innerText = label.innerText.split(' (')[0] + ` (${words} words)`;
    }
    document.querySelectorAll('.word_count textarea').forEach((t) => {
        t.addEventListener('input', wordCountHandler);
    })
}