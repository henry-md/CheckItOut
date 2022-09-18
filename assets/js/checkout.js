hideFullScreenDemo();

function showFullScreenDemo() {
    let el = document.getElementById('full-body-sleeve-fixed');
    el.style.top = "0";

    let bod = document.getElementsByTagName('body')[0];
    bod.classList.add('stop-scrolling');
}

function hideFullScreenDemo() {
    let el = document.getElementById('full-body-sleeve-fixed');
    el.style.top = "-9999px";

    let bod = document.getElementsByTagName('body')[0];
    bod.classList.remove('stop-scrolling');
}