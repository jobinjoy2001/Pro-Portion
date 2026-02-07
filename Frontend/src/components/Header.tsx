import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { Proportions, BookOpen, Pencil } from "lucide-react";

const Header = () => {
  const location = useLocation();

  const navItems = [
    { path: "/", label: "Home", icon: null },
    { path: "/learn", label: "Learn Mode", icon: BookOpen },
    { path: "/draw", label: "Draw Mode", icon: Pencil },
  ];

  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="fixed top-0 left-0 right-0 z-50 border-b border-border/30 bg-background/60 backdrop-blur-2xl"
    >
      <div className="container mx-auto flex items-center justify-between px-6 py-4">
        <Link to="/" className="flex items-center gap-3 group">
          <div className="relative">
            <Proportions className="h-7 w-7 text-primary transition-transform duration-300 group-hover:scale-110" />
            <div className="absolute inset-0 blur-lg bg-primary/30 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
          </div>
          <span className="text-xl font-bold tracking-tight">
            Pro<span className="text-primary">-</span>Portion
          </span>
        </Link>

        <nav className="flex items-center gap-1">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`relative flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-300 ${
                  isActive
                    ? "text-primary-foreground bg-primary/90"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/60"
                }`}
              >
                {item.icon && <item.icon className="h-4 w-4" />}
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </motion.header>
  );
};

export default Header;
