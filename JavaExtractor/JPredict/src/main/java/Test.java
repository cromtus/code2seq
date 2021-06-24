class Test {
    public static void test() {
        new Runnable() {
            @Override
            public void run() {
                System.out.println("Hello, world!");
            }
        }.run();
    }
}