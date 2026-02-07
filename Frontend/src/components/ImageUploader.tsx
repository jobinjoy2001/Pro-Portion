import { useCallback, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Image as ImageIcon, X } from "lucide-react";

interface ImageUploaderProps {
  onFileSelect: (file: File) => void;
  isProcessing: boolean;
}

const ImageUploader = ({ onFileSelect, isProcessing }: ImageUploaderProps) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string>("");

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target?.result as string);
      reader.readAsDataURL(file);
      onFileSelect(file);
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const clearPreview = () => {
    setPreview(null);
    setFileName("");
  };

  return (
    <div className="w-full">
      <AnimatePresence mode="wait">
        {preview ? (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="relative glass-card overflow-hidden"
          >
            <img
              src={preview}
              alt="Upload preview"
              className="w-full max-h-[400px] object-contain rounded-2xl"
            />
            <div className="absolute top-4 right-4 flex gap-2">
              <button
                onClick={clearPreview}
                disabled={isProcessing}
                className="p-2 rounded-xl bg-background/80 backdrop-blur-sm border border-border/50 text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-background/90 to-transparent">
              <p className="text-sm text-muted-foreground truncate">{fileName}</p>
            </div>
          </motion.div>
        ) : (
          <motion.label
            key="upload"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            htmlFor="image-upload"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={`upload-zone flex flex-col items-center justify-center py-20 px-8 cursor-pointer ${
              isDragging ? "upload-zone-active" : ""
            }`}
          >
            <div className="flex flex-col items-center gap-4">
              <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center">
                {isDragging ? (
                  <ImageIcon className="h-8 w-8 text-primary" />
                ) : (
                  <Upload className="h-8 w-8 text-primary" />
                )}
              </div>
              <div className="text-center">
                <p className="text-lg font-semibold mb-1">
                  {isDragging ? "Drop your image here" : "Upload a portrait"}
                </p>
                <p className="text-sm text-muted-foreground">
                  Drag & drop or click to browse · JPG, PNG supported
                </p>
              </div>
            </div>
            <input
              id="image-upload"
              type="file"
              accept="image/jpeg,image/png"
              onChange={handleInputChange}
              className="hidden"
            />
          </motion.label>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ImageUploader;
